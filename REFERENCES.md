# Scientific Articles
We encourage you to also check out work by the group behind 
GluonTS. They are grouped according to topic and ordered 
chronographically.

## Methods
A number of the below methods are available in GluonTS.

[A multivariate forecasting model](https://arxiv.org/abs/1910.03002)
```
@inproceedings{salinas2019high,
	Author = {Salinas, David and Bohlke-Schneider, Michael and Callot, Laurent and Gasthaus, Jan},
	Booktitle = {Advances in Neural Information Processing Systems},
	Title = {High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes},
	Year = {2019}
}
```

[Deep Factor models, a global-local forecasting method.](http://proceedings.mlr.press/v97/wang19k.html)
```
@inproceedings{wang2019deep,
	Author = {Wang, Yuyang and Smola, Alex and Maddix, Danielle and Gasthaus, Jan and Foster, Dean and Januschowski, Tim},
	Booktitle = {International Conference on Machine Learning},
	Pages = {6607--6617},
	Title = {Deep factors for forecasting},
	Year = {2019}
}
```
[DeepAR, an RNN-based probabilistic forecasting model](https://arxiv.org/abs/1704.04110)
```
@article{flunkert2019deepar,
	Author = {Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Tim Januschowski},
	Journal = {International Journal of Forecasting},
	Title = {DeepAR: Probabilistic forecasting with autoregressive recurrent networks},
	Year = {2019}
}
```
[A flexible way to model probabilistic forecasts via spline quantile forecasts.](http://proceedings.mlr.press/v89/gasthaus19a.html)
```
@inproceedings{gasthaus2019probabilistic,
	Author = {Gasthaus, Jan and Benidis, Konstantinos and Wang, Yuyang and Rangapuram, Syama Sundar and Salinas, David and Flunkert, Valentin and Januschowski, Tim},
	Booktitle = {The 22nd International Conference on Artificial Intelligence and Statistics},
	Pages = {1901--1910},
	Title = {Probabilistic Forecasting with Spline Quantile Function RNNs},
	Year = {2019}
}
```
[Using RNNs to parametrize State Space Models.](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting)
```
@inproceedings{rangapuram2018deep,
	Author = {Rangapuram, Syama Sundar and Seeger, Matthias W and Gasthaus, Jan and Stella, Lorenzo and Wang, Yuyang and Januschowski, Tim},
	Booktitle = {Advances in Neural Information Processing Systems},
	Pages = {7785--7794},
	Title = {Deep state space models for time series forecasting},
	Year = {2018}
}
```
[A scalable state space model. Note that code for this model
is currently not available in GluonTS.](https://papers.nips.cc/paper/6313-bayesian-intermittent-demand-forecasting-for-large-inventories)
```
@inproceedings{seeger2016bayesian,
	Author = {Seeger, Matthias W and Salinas, David and Flunkert, Valentin},
	Booktitle = {Advances in Neural Information Processing Systems},
	Pages = {4646--4654},
	Title = {Bayesian intermittent demand forecasting for large inventories},
	Year = {2016}
}
```



## Tutorials
Tutorials are available in bibtex and with accompanying material,
 in particular slides, linked from below.
 
### KDD 2019
[paper](https://dl.acm.org/citation.cfm?id=3332289) 
[slides](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
```
@inproceedings{faloutsos19forecasting,
  author    = {Faloutsos, Christos and
               Flunkert, Valentin and
               Gasthaus, Jan and
               Januschowski, Tim and
               Wang, Yuyang},
  title     = {Forecasting Big Time Series: Theory and Practice},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019.},
  year      = {2019}
  }
```
### SIGMOD 2019
[paper](https://dl.acm.org/citation.cfm?id=3314033&dl=ACM&coll=DL)
[supporting material](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
```
@inproceedings{faloutsos2019classical,
 author = {Faloutsos, Christos and Gasthaus, Jan and Januschowski, Tim and Wang, Yuyang},
 title = {Classical and Contemporary Approaches to Big Time Series Forecasting},
 booktitle = {Proceedings of the 2019 International Conference on Management of Data},
 series = {SIGMOD '19},
 publisher = {ACM},
 address = {New York, NY, USA},
 year = {2019}
} 
```
### VLDB 2018
[paper](http://www.vldb.org/pvldb/vol11/p2102-faloutsos.pdf)
[supporting material](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)
```
@article{faloutsos2018forecasting,
	Author = {Faloutsos, Christos and Gasthaus, Jan and Januschowski, Tim and Wang, Yuyang},
	Journal = {Proceedings of the VLDB Endowment},
	Number = {12},
	Pages = {2102--2105},
	Title = {Forecasting big time series: old and new},
	Volume = {11},
	Year = {2018}
}
```

## General audience
An overview of forecasting libraries in Python.
[paper to appear](https://foresight.forecasters.org/wp-content/uploads/Foresight_Issue55_cumTOC.pdf)
```
@article{januschowski19open,
  title={Open-Source Forecasting Tools in Python},
  author={Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang},
  journal={Foresight: The International Journal of Applied Forecasting},
  year={2019}
}
```
[A commentary on the M4 competition and its classification of the participating methods 
into 'statistical' and 'ML' methods. The article proposes alternative criteria.](https://www.sciencedirect.com/science/article/pii/S0169207019301529)
```
@article{januschowski19criteria,
title = {Criteria for classifying forecasting methods},
author = {Januschowski, Tim and Gasthaus, Jan and  Wang, Yuyang and Salinas, David and Flunkert, Valentin and Bohlke-Schneider, Michael and Callot, Laurent},
journal = {International Journal of Forecasting},
year = {2019}
}
```
[The business forecasting problem landscape can be divided into 
strategic, tactical and operational forecasting problems.](https://foresight.forecasters.org/product/foresight-issue-53/)
```
@article{januschowski18a,
  title={A Classification of Business Forecasting Problems},
  author={Januschowski, Tim and Kolassa, Stephan},
  journal={Foresight: The International Journal of Applied Forecasting},
  year={2019},
  volume={52}, 
  pages={36-43}
}
```
A two-part article introducing deep learning for forecasting.
[part 2](https://foresight.forecasters.org/product/foresight-issue-52/)
[part 1](https://foresight.forecasters.org/product/foresight-issue-51/)
```
@article{januschowski18deep2,
title = {Deep Learning for Forecasting: Current Trends and Challenges},
journal = {Foresight: The International Journal of Applied Forecasting},
year = {2018},
author = {Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang and Rangapuram, Syama Sundar and Callot, Laurent},
volume = {51}, 
pages = {42-47}
}
```
```
@article{januschowski18deep,
  title = {Deep Learning for Forecasting},
  author = {Januschowski, Tim and Gasthaus, Jan and Wang, Yuyang and Rangapuram, Syama and Callot, Laurent},
  journal = {Foresight},
  year = {2018}
}
```

## System Aspects
[A large-scale retail forecasting system.](http://www.vldb.org/pvldb/vol10/p1694-schelter.pdf)
```
@article{bose2017probabilistic,
	Author = {B{\"o}se, Joos-Hendrik and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim and Lange, Dustin and Salinas, David and Schelter, Sebastian and Seeger, Matthias and Wang, Yuyang},
	Journal = {Proceedings of the VLDB Endowment},
	Number = {12},
	Pages = {1694--1705},
	Title = {Probabilistic demand forecasting at scale},
	Volume = {10},
	Year = {2017}
}
```