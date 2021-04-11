# [Re] Spatial-Adaptive Network for Single Image Denoising

This repository is the re-production implementation of [Spatial-Adaptive Network for Single Image Denoising](https://arxiv.org/abs/2001.10291) by [Meng Chang](https://github.com/JimmyChame), Qi Li, Huajun Feng and Zhihai Xu in the scope of [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).

For more details please visit [our reproducibilty report on OpenReview](https://openreview.net/forum?id=yiAI9QN9nYt).

#

![][birds]
![][keys]

> **[Re] Spatial-Adaptive Network for Single Image Denoising**<br>
> Sami Menteş, Furkan Kınlı, Barış Özcan, Furkan Kıraç <br> > _Accepted to ReScience C Journal 2020_ <br>
>
> **Abstract:** In this study, we present our results and experience during replicating the paper titled "Spatial-Adaptive Network for Single Image Denoising". This paper proposes novel spatial-adaptive denoising architecture for efficient noise removal by leveraging the deformable convolutions to adapt spatial information (i.e. edges and textures). We have implemented the model from scratch in PyTorch framework, and then have conducted real and synthetic noise experiments on the corresponding datasets. We have achieved to reproduce the results qualitatively and quantitatively.

## Citation

If you find our reproducibility work useful in your study, please cite our paper as ;

```
@inproceedings{
mente{\c{s}}2021re,
title={[Re] Spatial-Adaptive Network for Single Image Denoising},
author={Sami Mente{\c{s}} and Furkan K{\i}nl{\i} and Bar{\i}{\c{s}} {\"O}zcan and Furkan K{\i}ra{\c{c}}},
booktitle={ML Reproducibility Challenge 2020},
year={2021},
url={https://openreview.net/forum?id=yiAI9QN9nYt}
}
```

## Acknowledgments

The blocks are adopted from [the original repo of JimmyChame](https://github.com/JimmyChame/SADNet).

The BibTex of the original work:

```
@InProceedings{10.1007/978-3-030-58577-8_11,
author="Chang, Meng
and Li, Qi
and Feng, Huajun
and Xu, Zhihai",
editor="Vedaldi, Andrea
and Bischof, Horst
and Brox, Thomas
and Frahm, Jan-Michael",
title="Spatial-Adaptive Network for Single Image Denoising",
booktitle="Computer Vision -- ECCV 2020",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="171--187"
}
```

[birds]: images/results.png
[keys]: images/results2.png
