"""
docker entrypoint is "CMD bash /usr/bin/start.sh"
/usr/bin/start.sh will call "python3 ymir/start.py"

main() --> start() --> _run_training() | _run_mining() | _run_infer()

mining and infer task can run at the same time.
"""
import logging
import os
import os.path as osp
import subprocess
import sys
from typing import List

import cv2
from easydict import EasyDict as edict
from ymir_exc import dataset_reader, env, monitor, result_writer
from ymir_exc.util import (YmirStage, get_bool, get_merged_config,
                           get_ymir_process, write_ymir_training_result)

from models.experimental import attempt_download
from ymir.ymir_yolov5 import (YmirYolov5, convert_ymir_to_yolov5,
                              get_weight_file)


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    else:
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            mining_task_idx = 0
            infer_task_idx = 1
            task_num = 2
        else:
            mining_task_idx = 0
            infer_task_idx = 0
            task_num = 1

        # task_idx and task_num will change the percent for monitor
        if cfg.ymir.run_mining:
            _run_mining(cfg, mining_task_idx, task_num)
        if cfg.ymir.run_infer:
            _run_infer(cfg, infer_task_idx, task_num)

    return 0


def _run_training(cfg: edict) -> None:
    """
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    # 1. convert dataset
    out_dir: str = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    # 2. training model
    epochs: int = int(cfg.param.epochs)
    batch_size: int = int(cfg.param.batch_size)
    img_size: int = int(cfg.param.img_size)
    save_weight_file_num: int = int(cfg.param.get('save_weight_file_num', 1))
    args_options: str = cfg.param.get('args_options', '')
    gpu_id: str = str(cfg.param.gpu_id)
    gpu_count: int = len(gpu_id.split(',')) if gpu_id else 0
    port: int = int(cfg.param.get('port', 29500))
    sync_bn: bool = get_bool(cfg, 'sync_bn', False)
    workers: int = int(cfg.param.get('workers', 8))
    cfg_file: str = cfg.param.get('cfg_file', 'cfg/training/yolov7-tiny.yaml')
    hyp_file: str = cfg.param.get('hyp_file', 'data/hyp.scratch.tiny.yaml')
    cache_images: bool = get_bool(cfg, 'cache_images', True)
    exist_ok: bool = get_bool(cfg, 'exist_ok', True)

    weights: str = get_weight_file(cfg)
    if not weights:
        # download pretrained weight from https://github.com/WongKinYiu/yolov7
        downloaded_model_name = osp.splitext(osp.basename(cfg_file))[0]
        weights = f"{downloaded_model_name}.pt"
        attempt_download(weights)

    models_dir: str = cfg.ymir.output.models_dir
    project: str = osp.dirname(models_dir)
    name: str = osp.basename(models_dir)

    commands: List[str] = ['python3']
    if gpu_count == 0:
        device = 'cpu'
    else:
        device = gpu_id
        if gpu_count > 1:
            commands.extend(
                f'-m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port}'
                .split())

    if osp.basename(cfg_file) in [
            'yolov7.yaml', 'yolov7-tiny.yaml', 'yolov7x.yaml'
    ]:
        commands.append('train.py')
    else:
        commands.append('train_aux.py')

    commands.extend([
        '--epochs',
        str(epochs), '--batch-size',
        str(batch_size), '--data', f'{out_dir}/data.yaml', '--workers',
        str(workers), '--project', project, '--name', name, '--cfg', cfg_file,
        '--hyp', hyp_file, '--weights', weights, '--img-size',
        str(img_size),
        str(img_size), '--device', device
    ])

    if gpu_count > 1 and sync_bn:
        commands.append("--sync-bn")

    if save_weight_file_num <= 1:
        commands.append("--nosave")
    else:
        save_period = max(1, epochs // save_weight_file_num)
        commands.extend(['--save_period', str(save_period)])

    if cache_images:
        commands.append("--cache-images")

    if exist_ok:
        commands.append("--exist-ok")

    if args_options:
        commands.extend(args_options.split())

    logging.info(f'start training: {commands}')

    subprocess.run(commands, check=True)

    # save other files in output directory, map50 will use the best one.
    write_ymir_training_result(cfg, map50=0, files=[], id='last')
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    del cfg
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS,
                                 p=1.0,
                                 task_idx=task_idx,
                                 task_num=task_num))

    command = 'python3 ymir/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.POSTPROCESS,
                                 p=1.0,
                                 task_idx=task_idx,
                                 task_num=task_num))


def _run_infer(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS,
                                 p=1.0,
                                 task_idx=task_idx,
                                 task_num=task_num))

    N = dataset_reader.items_count(env.DatasetType.CANDIDATE)
    infer_result = {}
    model = YmirYolov5(cfg)
    idx = -1

    monitor_gap = max(1, N // 100)
    for asset_path, _ in dataset_reader.item_paths(
            dataset_type=env.DatasetType.CANDIDATE):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        infer_result[asset_path] = result
        idx += 1

        if idx % monitor_gap == 0:
            percent = get_ymir_process(stage=YmirStage.TASK,
                                       p=idx / N,
                                       task_idx=task_idx,
                                       task_num=task_num)
            monitor.write_monitor_logger(percent=percent)

    result_writer.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(
        percent=get_ymir_process(stage=YmirStage.PREPROCESS,
                                 p=1.0,
                                 task_idx=task_idx,
                                 task_num=task_num))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    # https://github.com/protocolbuffers/protobuf/issues/10051
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start())
