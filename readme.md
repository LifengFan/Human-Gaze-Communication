
# ICCV2019 - Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning

Introduction
----

The project is described in our paper [Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning](https://lifengfan.github.io/files/iccv19/ICCV19_Gaze_Communication.pdf) (ICCV2019).   

This paper addresses a new problem of understanding human gaze communication in social videos from both atomic-level and event-level, which is significant for studying human social interactions. To tackle this novel and challenging problem, we contribute a large-scale video dataset, VACATION, which covers diverse daily social scenes and gaze communication behaviors with complete annotations of objects and human faces, human attention, and communication structures and labels in both atomic-level and event-level. Together with VACATION, we propose a spatiotemporal graph neural network to explicitly represent the diverse gaze interactions in the social scenes and to infer atomic-level gaze communication by message passing. We further propose an event network with encoderdecoder structure to predict the event-level gaze communication. Our experiments demonstrate that the proposed model improves various baselines significantly in predicting the atomic-level and event-level gaze communications.
![](https://github.com/LifengFan/Human-Gaze-Communication/blob/master/doc/teaser.jpg)  

Dataset
----

Please fill this [online form](https://forms.gle/uSeLsShXefyHjAgCA) to get a copy of the dataset and annotation. We will get back to you in a day or two.

Demo
----

Here is a [demo](https://vimeo.com/985557163?share=copy) to show more dynamic results.

Citation
----

Please cite our paper if you find the project and the dataset useful:


```
@inproceedings{fan2019understanding,
  title     = {Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning},
  author    = {Lifeng Fan and Wenguan Wang and Siyuan Huang and Xinyu Tang and Song-Chun Zhu},
  year      = {2019},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)}
}
```
