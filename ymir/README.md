# ymir-yolov5

```
docker pull youdaoyzbx/ymir-executor:ymir1.1.0-yolov7-cu111-tmi

docker build -t ymir/executor:yolov7-cu111-tmi --build-arg YMIR=1.1.0 -f ymir/docker/cuda111.dockerfile .
```


## change log
- add `ymir` folder
- modify `train.py` to write `monitor.txt` and `result.yaml`
- modify `utils/datasets.py` to support ymir dataset format


## support ymir dataset format
- use `convert_ymir_to_yolov5` to generate `data.yaml`

- modify `utils/datasets.py` load `train.tsv`, `val.tsv`
```
# support ymir index file `train.tsv` and `val.tsv`
# class LoadImagesAndLabels(Dataset):
f = []  # image files
img2label_map = dict()  # map image files to label files
for p in path if isinstance(path, list) else [path]:
    p = Path(p)  # os-agnostic
    if p.is_file():  # file
        with open(p, 'r') as t:
            t = t.read().strip().splitlines()
            for x in t:
                # x = f'{image_path}\t{label_path}\n'
                image_path, label_path = x.split()
                f.append(image_path)
                img2label_map[image_path] = label_path
    else:
        raise Exception(f'{prefix}{p} is not valid ymir index file')

# get label file path from image file path
def img2label_paths(img_paths, img2label_map={}):
    return [img2label_map[img] for img in img_paths]

self.label_files = img2label_paths(self.img_files, img2label_map)

# support ymir label file
# convert ymir (xyxy) to yolov5 bbox format normalized (xc,yc,w,h)
# def cache_labels()
l_ymir = np.array(l, dtype=np.float32)
l = l_ymir.copy()
width, height = imagesize.get(im_file)
l[:,1:5:2] /= width # normalize x1,x2
l[:,2:5:2] /= height # normalize y1,y2
l[:,1] = (l_ymir[:,1] + l_ymir[:,3])/2/width
l[:,2] = (l_ymir[:,2] + l_ymir[:,4])/2/height
l[:,3] = (l_ymir[:,3] - l_ymir[:,1])/width
l[:,4] = (l_ymir[:,4] - l_ymir[:,2])/height
```

## write monitor and training results

modify `train.py`
- import functions
```
from ymir.ymir_yolov5 import get_merged_config, YmirStage, get_ymir_process
from ymir_exc import monitor
from ymir_exc import result_writer as rw
```

- write tensorboard results
```
# change tensorboard writer log_dir
ymir_cfg = get_merged_config()
tb_writer = SummaryWriter(ymir_cfg.ymir.output.tensorboard_dir)  # Tensorboard
```

- monitor training process
```
# for each epoch
monitor_gap = max(1, (epochs - start_epoch)//1000)
if rank in [-1, 0] and epoch % monitor_gap == 0:
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=(epoch-start_epoch)/(epochs-start_epoch)))
```

- write `result.yaml` to save model weights and map50
    - optional: modify `utils/metrics.py` fitness() to save best map50
    ```
    def fitness(x):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 1.0, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (x[:, :4] * w).sum(1)
    ```
```
if (epoch + 1) % opt.save_period == 0:
    epoch_weight_file = wdir / 'epoch_{:03d}.pt'.format(epoch)
    torch.save(ckpt, epoch_weight_file)
    write_ymir_training_result(ymir_cfg, map50=float(results[2]), id=str(epoch), files=[str(epoch_weight_file)])
```

- modify `start.py` save other output files
```
# save other files in output directory
write_ymir_training_result(cfg, map50=0, id='last', files=[])
# if task done, write 100% percent log
monitor.write_monitor_logger(percent=1.0)
```

## infer and mining
- view `ymir/start.py` "_run_infer()" for infer
- view `ymir/start.py` "_run_mining()" for mining
