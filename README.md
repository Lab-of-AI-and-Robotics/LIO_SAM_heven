# LIO-SAM

[original repo](https://github.com/TixiaoShan/LIO-SAM)

## Manual
- [without docker](https://github.com/Lab-of-AI-and-Robotics/Lair_Code_Implementation_Manual/blob/main/manual/How_to_implement_LIO-SAM_and_FAST-LIO.md)
- [with docker](https://github.com/Lab-of-AI-and-Robotics/Lair_Code_Implementation_Manual/blob/main/manual/LIO_SAM.md)

## Saving map

```bash
rosservice call /lio_sam/save_map 0.2 "/data/maps/"
```

## 보다 강건한 시스템을 위해 수정 가능한 파라미터들
- relocalize success score
- 