
*frame* - turn a directory of videos into directory of frames

*extract* - extract a directory of frames into faces with given size 

### frame.py
``` bash
python3 -u frame.py --vdir {input video directory} --fdir {output directory of frames} --skip {skip how many frames, default=50}
```

### extract.py
``` bash
python3 -u extract.py --fdir {input frame directory} --pdir {output of faces} --size {image of face = size*size, default=192}
```
