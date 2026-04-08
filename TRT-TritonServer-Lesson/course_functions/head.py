import math
from typing import Tuple

import torch
from torch import Tensor, Value
import torch.nn.functional as F
from ultralytics.nn.modules import Detect, OBB, Pose, Segment
from ultralytics.utils.tal import make_anchors

__all__ = ["UltralyticsDetect", "UltralyticsOBB", "UltralyticsSegment", "UltralyticsPose"]


class EfficientNMS_TRT(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1, # 没有背景类别
            score_activation_i=1,  # 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            class_agnostic_i=1,    # 执行类无关的 NMS
            box_coding_i=1,        # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


class EfficientRotatedNMS_TRT(torch.autograd.Function):
    """RotatedNMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 5, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientRotatedNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1, # 没有背景类别
            score_activation_i=1,  # 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            class_agnostic_i=1,    # 执行类无关的 NMS
            box_coding_i=1,        # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


class EfficientIdxNMS_TRT(torch.autograd.Function):
    """NMS with Index block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: float = 100,
    ) -> Tuple[Value, Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientIdxNMS_TRT',
            boxes,
            scores,
            outputs=5,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1, # 没有背景类别
            score_activation_i=1,  # 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            class_agnostic_i=1,    # 执行类无关的 NMS
            box_coding_i=1,        # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


"""
===============================================================================
            Ultralytics Detect head for detection models
===============================================================================
"""


class UltralyticsDetect(Detect):
    """Ultralytics 检测头，用于检测模型"""

    max_det = 100  # 最大检测框数量
    iou_thres = 0.65  # IoU阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """返回预测的边界框和类别概率"""
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)] # 拼接特征图
        dbox, cls = self._inference(x) # 解码边界框和类别概率

        # 使用 EfficientNMS_TRT 插件
        return EfficientNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

    def _inference(self, x):
        """基于多尺度特征图解码预测的边界框和类别概率"""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接特征图
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚点
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分离边界框和类别
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # 解码边界框

        return dbox, cls  # 返回解码后的边界框和类别概率

class UltralyticsOBB(OBB):
    """Ultralytics 旋转边界框检测头，用于旋转检测模型"""

    max_det = 100  # 最大检测框数量
    iou_thres = 0.65  # IoU阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """返回预测的旋转边界框和类别概率"""
        bs = x[0].shape[0]  # 批量大小
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB 角度 logits
        # NOTE: 将 angle 设置为属性，以便 decode_bboxes 可以使用它
        angle = (angle.sigmoid() - 0.25) * math.pi  # 将角度归一化到 [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        self.angle = angle # 不要删除，用于 OBB decode_bboxes

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)] # 拼接特征图
        dbox, cls = self._inference(x) # 解码边界框和类别概率
        rotated_box = torch.cat([dbox, angle], 1).transpose(1, 2) # 拼接旋转边界框

        # 使用 EfficientRotatedNMS_TRT 插件
        return EfficientRotatedNMS_TRT.apply(
            rotated_box,
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

    def _inference(self, x):
        """基于多尺度特征图解码预测的边界框和类别概率"""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接特征图
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚点
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分离边界框和类别
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # 解码边界框

        return dbox, cls  # 返回解码后的边界框和类别概率


class UltralyticsSegment(Segment):
    """Ultralytics 分割头，用于分割模型"""

    max_det = 100  # 最大检测框数量
    iou_thres = 0.65  # IoU阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """返回预测的边界框、类别概率和掩码系数"""
        p = self.proto(x[0])  # 掩码原型
        bs, _, mask_h, mask_w = p.shape # 批量大小和掩码尺寸
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2).permute(0, 2, 1)  # 掩码系数

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)] # 拼接特征图
        dbox, cls = self._inference(x) # 解码边界框和类别概率

        # 使用 EfficientIdxNMS_TRT 插件
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

        # 根据检测索引选择掩码系数，掩码系数 * 掩码原型 得到 分割掩码
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        selected_mc = mc[bs_indices, det_indices]
        det_masks = torch.einsum('b d n, b n h w -> b d h w', selected_mc, p).sigmoid()
        
        """
        使用 torch.einsum 代替以下代码：

        masks_protos = p.view(bs, self.nm, mask_h * mask_w)
        det_masks = torch.matmul(selected_mc, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)
        """

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8), # 上采样并二值化掩码
        )

    def _inference(self, x):
        """基于多尺度特征图解码预测的边界框和类别概率"""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接特征图
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚点
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分离边界框和类别
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # 解码边界框

        return dbox, cls  # 返回解码后的边界框和类别概率


class UltralyticsPose(Pose):
    """Ultralytics 姿态估计头，用于关键点检测模型"""

    max_det = 100  # 最大检测框数量
    iou_thres = 0.65  # IoU阈值
    conf_thres = 0.25  # 置信度阈值

    def forward(self, x):
        """返回预测的边界框、类别概率和关键点"""
        bs = x[0].shape[0]  # 批量大小
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w) 关键点预测

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)] # 拼接特征图
        dbox, cls = self._inference(x) # 解码边界框和类别概率

        # 使用 EfficientIdxNMS_TRT 插件
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

        pred_kpts = self.kpts_decode(bs, kpt).transpose(1, 2) # 解码关键点
        # 根据检测索引选择关键点
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        det_kpts = pred_kpts[bs_indices, det_indices].view(bs, self.max_det, *self.kpt_shape)

        return num_dets, det_boxes, det_scores, det_classes, det_kpts

    def _inference(self, x):
        """基于多尺度特征图解码预测的边界框和类别概率"""
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)  # 拼接特征图
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))  # 生成锚点
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)  # 分离边界框和类别
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides  # 解码边界框

        return dbox, cls  # 返回解码后的边界框和类别概率