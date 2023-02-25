# Credit to Justin Johnsons' EECS-598 course at the University of Michigan,
# from which this assignment is heavily drawn.
import math
from typing import Dict, List, Optional

import torch
#from a4_helper import *
from detection_utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from torchvision import models
from torchvision.models import feature_extraction


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:
        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32
    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

    
        # Initialize additional Conv layers for FPN.                   
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()
        for in_channels in dummy_out_shapes:
            inner_block = "c{}_conv1*1".format(in_channels[0][1])
            layer_block = "p{}_conv3*3".format(in_channels[0][1])
            inner_block_module = nn.Conv2d(in_channels[1][1], self.out_channels, 1)
            layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, padding = 1)

            self.fpn_params[inner_block] = inner_block_module
            self.fpn_params[layer_block] = layer_block_module

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        # Fill output FPN features (p3, p4, p5) using RegNet features (c3, c4, c5) and FPN conv layers created above.                    #
        p5_conv1_1 = self.fpn_params["c5_conv1*1"](backbone_feats["c5"])
        p5_upsampled_x = F.interpolate(p5_conv1_1, size=(backbone_feats["c4"].shape[2], backbone_feats["c4"].shape[3]), mode="nearest")
        p5_x = self.fpn_params["p5_conv3*3"](p5_conv1_1)
        
        p4_conv1_1 = self.fpn_params["c4_conv1*1"](backbone_feats["c4"])
        p4_x = p5_upsampled_x + p4_conv1_1
        p4_upsampled_x = F.interpolate(p4_x, size=(backbone_feats["c3"].shape[2], backbone_feats["c3"].shape[3]), mode="nearest")
        p4_x = self.fpn_params["p4_conv3*3"](p4_x)
        
        p3_conv1_1 = self.fpn_params["c3_conv1*1"](backbone_feats["c3"])
        p3_x = p3_conv1_1 + p4_upsampled_x
        p3_x = self.fpn_params["p3_conv3*3"](p3_x)
        
        fpn_feats["p3"] = p3_x
        fpn_feats["p4"] = p4_x
        fpn_feats["p5"] = p5_x

        return fpn_feats

class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355
    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()
        stem_cls = []
        stem_box = []

        conv = nn.Conv2d(in_channels, stem_channels[0], kernel_size = 3, stride = 1,
                    padding = 1, bias = True)
        nn.init.normal_(conv.weight, mean=0, std=0.01)
        nn.init.zeros_(conv.bias)
        stem_cls.append(conv)
        stem_cls.append(nn.ReLU())
        # box
        conv2 = nn.Conv2d(in_channels, stem_channels[0], kernel_size = 3, stride = 1,
                    padding = 1, bias = True)
        nn.init.normal_(conv2.weight, mean=0, std=0.01)
        nn.init.zeros_(conv2.bias)
        stem_box.append(conv2)
        stem_box.append(nn.ReLU())
        # middle layers
        for i in range(len(stem_channels)-1):
            # cls
            conv = nn.Conv2d(stem_channels[i], stem_channels[i+1], kernel_size = 3, stride = 1,
                            padding = 1, bias = True)
            nn.init.normal_(conv.weight, mean=0, std=0.01)
            nn.init.zeros_(conv.bias)
            stem_cls.append(conv)
            stem_cls.append(nn.ReLU())
            # box
            conv2 = nn.Conv2d(stem_channels[i], stem_channels[i+1], kernel_size = 3, stride = 1,
                            padding = 1, bias = True)
            nn.init.normal_(conv2.weight, mean=0, std=0.01)
            nn.init.zeros_(conv2.bias)
            stem_box.append(conv2)
            stem_box.append(nn.ReLU())

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # Replace "pass" statement with your code
        cls_conv = nn.Conv2d(stem_channels[-1], num_classes, 3, padding=1)
        nn.init.normal_(cls_conv.weight, mean=0, std=0.01)
        nn.init.zeros_(cls_conv.bias)
        self.pred_cls = cls_conv

        box_conv = nn.Conv2d(stem_channels[-1], 4, 3, padding=1)
        nn.init.normal_(box_conv.weight, mean=0, std=0.01)
        nn.init.zeros_(box_conv.bias)
        self.pred_box = box_conv

        ctr_conv = nn.Conv2d(stem_channels[-1], 1, 3, padding=1)
        nn.init.normal_(ctr_conv.weight, mean=0, std=0.01)
        nn.init.zeros_(ctr_conv.bias)
        self.pred_ctr = ctr_conv

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).
        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).
        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}
        level = ["p3", "p4", "p5"]
        for l in level:
            class_logits[l] = self.pred_cls(self.stem_cls(feats_per_fpn_level[l])).flatten(start_dim=2).permute(0, 2, 1)
            boxreg_deltas[l] = self.pred_box(self.stem_box(feats_per_fpn_level[l])).flatten(start_dim=2).permute(0, 2, 1)
            centerness_logits[l] = self.pred_ctr(self.stem_box(feats_per_fpn_level[l])).flatten(start_dim=2).permute(0, 2, 1)

        return [class_logits, boxreg_deltas, centerness_logits]

class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector
    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        self.backbone = None
        self.pred_net = None
        # Replace "pass" statement with your code
        self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.pred_net = FCOSPredictionNetwork(
            num_classes=num_classes, 
            in_channels=fpn_channels, 
            stem_channels=stem_channels
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.
        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        # Replace "pass" statement with your code
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)

        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        device = images.device
        
        # Get shapes of each FPN level feature map.
        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in fpn_feats.items()
        }

        locations_per_fpn_level = get_fpn_location_coords(
                fpn_feats_shapes, self.backbone.fpn_strides, device=device
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        for gt_boxes_i in gt_boxes:
            matched_gt_boxes.append(fcos_match_locations_to_gt(
                    locations_per_fpn_level, 
                    self.backbone.fpn_strides,
                    gt_boxes_i
            ))

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        # Replace "pass" statement with your code
        #for i, gt_boxes_i in enumerate(gt_boxes):
        for matched_gt_boxes_i in matched_gt_boxes:
            gt_delta_per_fpn_level = {}
            for level_name, locations in locations_per_fpn_level.items():
            #for level_name, locations in matched_gt_boxes[i].items():
                #import pdb; pdb.set_trace()
                gt_delta_per_fpn_level[level_name] = fcos_get_deltas_from_locations(
                        locations, 
                        matched_gt_boxes_i[level_name], 
                        stride=self.backbone.fpn_strides[level_name]
                )
                #import pdb; pdb.set_trace()
            
            matched_gt_deltas.append(gt_delta_per_fpn_level)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None

        # Replace "pass" statement with your code
        
        # loss_cls
        # make a one-hot vectors
        gt_class_tensor = matched_gt_boxes[:, :, 4].clone()
        gt_class_tensor = gt_class_tensor.to(torch.int64)
        gt_class_tensor += 1
        #import pdb; pdb.set_trace()
        gt_classes = F.one_hot(gt_class_tensor, num_classes=self.num_classes+1) #(B, N, 21)
        #import pdb; pdb.set_trace()
        gt_classes = gt_classes.to(matched_gt_boxes.dtype)
        gt_classes = gt_classes[:,:,1:] # ignore the first index indication background class
        #import pdb; pdb.set_trace()
        loss_cls = sigmoid_focal_loss(
            inputs=pred_cls_logits, targets=gt_classes
        )
        #import pdb; pdb.set_trace()
        # loss_box
        loss_box = 0.25 * F.l1_loss(
            pred_boxreg_deltas, matched_gt_deltas, reduction="none"
        )
        # no loss for background:
        loss_box[matched_gt_deltas < 0] *= 0.0

        # calculate centerness loss.
        B = matched_gt_deltas.shape[0]
        N = matched_gt_deltas.shape[1]
        gt_centerness = torch.zeros(B, N, device=device)
        for i, matched_gt_deltas_i in enumerate(matched_gt_deltas):
            gt_centerness[i] = fcos_make_centerness_targets(matched_gt_deltas_i)
        
        gt_centerness = gt_centerness.view(gt_centerness.shape[0], gt_centerness.shape[1], 1)
        #import pdb; pdb.set_trace()
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits, gt_centerness, reduction="none"
        )
        # No loss for background:
        loss_ctr[gt_centerness < 0] *= 0.0

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.
        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.
                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).
                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1:
            # Replace "pass" statement with your code
            device = images.device
            N = level_pred_scores.shape[0]
            idx1 = torch.arange(N, device=device)
            predict_idx = torch.argmax(level_pred_scores, dim=1)
            level_pred_scores = level_pred_scores[idx1, predict_idx]
            #import pdb; pdb.set_trace()

            # Step 2:
            # Replace "pass" statement with your code
            mask = (level_pred_scores > test_score_thresh)
            level_pred_scores = level_pred_scores[mask]
            level_pred_classes = predict_idx[mask]
            #import pdb; pdb.set_trace()


            # Step 3:
            # Replace "pass" statement with your code
            valid_deltas = level_deltas[mask]
            valid_locations = level_locations[mask]
            level_pred_boxes = fcos_apply_deltas_to_locations(
                valid_deltas, 
                valid_locations, 
                stride=self.backbone.fpn_strides[level_name]
            )
            #import pdb; pdb.set_trace()

            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "pass" statement with your code
            _, _, H, W = images.shape
            #idx1 = torch.arange(level_pred_boxes.shape[0], device=device)
            X = level_pred_boxes[:, 0:4:2]
            Y = level_pred_boxes[:, 1:5:2]
            #import pdb; pdb.set_trace()
            X = torch.clamp(X, min=0, max=W)
            Y = torch.clamp(Y, min=0, max=H)
            level_pred_boxes[:, 0:4:2] = X
            level_pred_boxes[:, 1:5:2] = Y
            #import pdb; pdb.set_trace()
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )