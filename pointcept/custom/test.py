import os
import time
import json
import numpy as np
from uuid import uuid4
from collections import OrderedDict
import torch
import torch.utils.data

from pointcept.engines.defaults import create_ddp_model
from pointcept.engines.test import TESTERS, TesterBase
import pointcept.utils.comm as comm
from pointcept.models import build_model
from pointcept.utils.misc import AverageMeter, make_dirs
from .misc import process_label, process_instance

NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

@TESTERS.register_module()
class InsSegTester(TesterBase):
    def __init__(
            self, cfg, model=None, segment_ignore_index=(-1,0,1), 
            semantic_ignore_index=(-1,), instance_ignore_index=-1, 
            test_loader=None, verbose=False,    
        ):
        super().__init__(cfg, model, test_loader, verbose)
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index=semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.valid_class_names = [
            self.cfg.data.names[i]
            for i in range(self.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def test(self):
        assert self.test_loader.batch_size == 1
        self.logger.info(">>>>>>>>>>>>>>>> Start Test >>>>>>>>>>>>>>>>")
        if comm.is_main_process():
            print("valid_class_names:", self.valid_class_names, len(self.valid_class_names))
            print("class2id: ", self.test_loader.dataset.class2id)

        batch_time = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        if "test" in self.cfg.data.test.split:
            # create submit folder only on main process
            if (
                self.cfg.data.test.type == "ScanNetSpDataset" or 
                self.cfg.data.test.type == "ScanNetppDataset" or
                self.cfg.data.test.type == "ScanNetPPSpDataset"
            ) and comm.is_main_process():
                make_dirs(os.path.join(save_path, "submit"))
                make_dirs(os.path.join(save_path, "submit", "predicted_masks"))
        else:
            # create val folder only on main process
            if (
                self.cfg.data.test.type == "ScanNetSpDataset" or 
                self.cfg.data.test.type == "ScanNetppDataset" or
                self.cfg.data.test.type == "ScanNetPPSpDataset"
            ) and comm.is_main_process():
                make_dirs(os.path.join(save_path, "val"))
                make_dirs(os.path.join(save_path, "val", "predicted_masks"))
                
        comm.synchronize()
        scenes = []
        # inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            data = data_dict.pop("data_dict")
            input_dict = data
            segment = data_dict.pop("segment")
            instance = data_dict.pop("instance")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                self.logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path, allow_pickle=True)
                pred = pred.item()
            else:
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred = self.model(input_dict)
                np.save(pred_save_path, pred)

            if "test" in self.cfg.data.test.split:
                batch_time.update(time.time() - end)
                self.logger.info(
                    "Test: {} [{}/{}]-Num {} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                    )
                )

                if self.cfg.data.test.type == "ScanNetSpDataset":
                    f = open(os.path.join(save_path, "submit", "{}.txt".format(data_name)), "w")
                    for i in range(len(pred["pred_classes"])):
                        label_id = NYU_ID[pred["pred_classes"][i]]
                        conf = pred["pred_scores"][i]
                        f.write(f'predicted_masks/{data_name}_{i:03d}.txt {label_id} {conf:.4f}\n')
                        mask_path = os.path.join(save_path, "submit", "predicted_masks", "{}_{:03d}.txt".format(data_name, i))
                        np.savetxt(mask_path, pred["pred_masks"][i].astype(np.uint8), fmt="%d")
                
                if (
                    self.cfg.data.test.type == "ScanNetppDataset" or
                    self.cfg.data.test.type == "ScanNetPPSpDataset"
                ):
                    f = open(os.path.join(save_path, "submit", "{}.txt".format(data_name)), "w")
                    for i in range(len(pred["pred_classes"])):
                        label_id = self.test_loader.dataset.class2id[pred["pred_classes"][i]]
                        conf = pred["pred_scores"][i]
                        f.write(f'predicted_masks/{data_name}_{i:03d}.json {label_id} {conf:.4f}\n')
                        inst_mask = rle_encode(pred["pred_masks"][i])
                        mask_path = os.path.join(save_path, "submit", "predicted_masks", "{}_{:03d}.json".format(data_name, i))
                        write_json(mask_path, inst_mask)
            else:
                loss = pred["loss"]
                batch_time.update(time.time() - end)
                self.logger.info(
                    "Test: {} [{}/{}]-Num {} "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Loss {loss:.4f} ".format(
                        data_name,
                        idx + 1,
                        len(self.test_loader),
                        segment.size,
                        batch_time=batch_time,
                        loss=loss.item()
                    )
                )

                assert "origin_instance" and "origin_segment" in input_dict.keys()
                instance = process_instance(
                    input_dict["origin_instance"].clone(),
                    input_dict["origin_segment"].clone(),
                    self.segment_ignore_index)
                segment = process_label(
                    input_dict["origin_segment"].clone(),
                    self.segment_ignore_index)

                gt_instances, pred_instance = self.associate_instances(
                    pred, segment, instance
                )
                scenes.append(dict(gt=gt_instances, pred=pred_instance))

                if self.cfg.data.test.type == "ScanNetSpDataset":
                    f = open(os.path.join(save_path, "val", "{}.txt".format(data_name)), "w")
                    for i in range(len(pred["pred_classes"])):
                        label_id = NYU_ID[pred["pred_classes"][i]]
                        conf = pred["pred_scores"][i]
                        f.write(f'predicted_masks/{data_name}_{i:03d}.txt {label_id} {conf:.4f}\n')
                        mask_path = os.path.join(save_path, "val", "predicted_masks", "{}_{:03d}.txt".format(data_name, i))
                        np.savetxt(mask_path, pred["pred_masks"][i].astype(np.uint8), fmt="%d")
                
                if (
                    self.cfg.data.test.type == "ScanNetppDataset" or
                    self.cfg.data.test.type == "ScanNetPPSpDataset"
                ):
                    f = open(os.path.join(save_path, "val", "{}.txt".format(data_name)), "w")
                    for i in range(len(pred["pred_classes"])):
                        label_id = self.test_loader.dataset.class2id[pred["pred_classes"][i]]
                        conf = pred["pred_scores"][i]
                        f.write(f'predicted_masks/{data_name}_{i:03d}.json {label_id} {conf:.4f}\n')
                        inst_mask = rle_encode(pred["pred_masks"][i])
                        mask_path = os.path.join(save_path, "val", "predicted_masks", "{}_{:03d}.json".format(data_name, i))
                        write_json(mask_path, inst_mask)
            torch.cuda.empty_cache()
        if "test" not in self.cfg.data.test.split:
            self.logger.info("Syncing ...")
            comm.synchronize()
            scenes_sync = comm.gather(scenes, dst=0)
            if comm.is_main_process():
                scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
                ap_scores = self.evaluate_matches(scenes)
                self.print_results(ap_scores)

    @staticmethod
    def collate_fn(batch):
        return batch
    
    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {}, best metric value {})".format(
                    self.cfg.weight, checkpoint["epoch"], checkpoint["best_metric_value"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model
    
    def print_results(self, ap_scores):
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_prec_50 = ap_scores["all_prec_50%"]
        all_rec_50 = ap_scores["all_rec_50%"]

        sep = ""
        col1 = ":"
        lineLen = 66
        self.logger.info("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>10}".format("AP") + sep
        line += "{:>10}".format("AP_50%") + sep
        line += "{:>10}".format("AP_25%") + sep
        line += "{:>10}".format("Prec_50%") + sep
        line += "{:>10}".format("Rec_50%") + sep
        self.logger.info(line)
        self.logger.info("#" * lineLen)

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            prec_50 = ap_scores["classes"][label_name]["prec50%"]
            rec_50 = ap_scores["classes"][label_name]["rec50%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>10.3f}".format(ap) + sep
            line += sep + "{:>10.3f}".format(ap_50) + sep
            line += sep + "{:>10.3f}".format(ap_25) + sep
            line += sep + "{:>10.3f}".format(prec_50) + sep
            line += sep + "{:>10.3f}".format(rec_50) + sep
            self.logger.info(line)
            
        self.logger.info("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>10.3f}".format(all_ap) + sep
        line += "{:>10.3f}".format(all_ap_50) + sep
        line += "{:>10.3f}".format(all_ap_25) + sep
        line += "{:>10.3f}".format(all_prec_50) + sep
        line += "{:>10.3f}".format(all_rec_50) + sep
        self.logger.info(line)
        self.logger.info("#" * lineLen)
    
        self.logger.info("<<<<<<<<<<<<<<<<< End Test <<<<<<<<<<<<<<<<<")
    
    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.semantic_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for name in self.valid_class_names:
            gt_instances[name] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.semantic_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.valid_class_names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for name in self.valid_class_names:
            pred_instances[name] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.semantic_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.valid_class_names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances
    
    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        pr_rc = np.zeros((2, len(self.valid_class_names), len(overlaps)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.semantic_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        #compute optimal precision and recall, based on f1_score
                        f1_score = 2 * precision * recall / (precision + recall + 0.0001)
                        f1_argmax = f1_score.argmax()
                        best_pr = precision[f1_argmax]
                        best_rc = recall[f1_argmax]

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        best_pr = 0
                        best_rc = 0
                    else:
                        ap_current = float("nan")
                        best_pr = float('nan')
                        best_rc = float('nan')
                    ap_table[di, li, oi] = ap_current
                    pr_rc[0, li, oi] = best_pr
                    pr_rc[1, li, oi] = best_rc
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores['all_prec_50%'] = np.nanmean(pr_rc[0, :, o50])
        ap_scores['all_rec_50%'] = np.nanmean(pr_rc[1, :, o50])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
            ap_scores["classes"][label_name]["prec50%"] = np.average(
                pr_rc[0, li, o50]
            )
            ap_scores["classes"][label_name]["rec50%"] = np.average(
                pr_rc[1, li, o50]
            )
        return ap_scores


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle

def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle['length']
    counts = rle['counts']
    s = counts.split()
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def write_json(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=4))

