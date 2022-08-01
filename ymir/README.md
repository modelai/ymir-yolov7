# ymir-yolov5

## change log
- add `ymir` folder
- modify `train.py` to write `monitor.txt` and `result.yaml`
- modify `utils/datasets.py` to support ymir dataset format


## ymir dataset format
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
