# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import pylab as pl

pl.rcParams["figure.figsize"] = 4, 4


# In[2]:


from GMMBase import *


# In[3]:


lamb = 0.5
xmax = 4 / lamb
n_sample = int(1e4)
n_clusters = 3
epochs = 1000


# In[4]:


x = np.random.RandomState(0).exponential(1.0 / lamb, n_sample)


# In[5]:


def Gaussian_pdf(ex):
    return np.exp(
        -0.5 * (ex - x.mean()) ** 2 / x.std() ** 2
        - 0.5 * np.log(2 * np.pi * x.std() ** 2)
    )


# In[6]:


ex = np.linspace(0, xmax)
pl.hist(x, ex, density=True, fill=False, histtype="step", label="empirical")
pl.plot(ex, lamb * np.exp(-lamb * ex), label="exponential")

pl.plot(ex, Gaussian_pdf(ex), label="Gaussian")
pl.errorbar(
    [x.mean()],
    Gaussian_pdf([x.mean()]),
    None,
    [x.std()],
    "x",
    capsize=2,
    label="Gaussian mode",
)

pl.legend(loc="upper right")
pl.xlim(xmax=xmax)


# # EM

# In[7]:


model = GMMModel(n_clusters, 1)
model.initialize()
trainer = GMMTrainer(model)

for epoch in range(100):
    trainer(x[:, None])
    print(epoch, model(mx.nd.array(x[:, None]))[0].mean().asscalar())


# In[8]:


mu_ = model.mu_.data().asnumpy()
kR_ = model.kR_.data().asnumpy()
cov_ = np.linalg.inv(kR_.swapaxes(1, 2) @ kR_)
s2_ = np.array([np.diag(c) for c in cov_])


# In[9]:


def mixture_pdf(ex):
    model = GMMModel(n_clusters, 1, mu_, kR_)
    model.initialize()
    log_marg = model(mx.nd.array(ex, dtype="float32"))[0]
    return log_marg.exp().asnumpy()


# In[10]:


ex = np.linspace(0, xmax)
pl.hist(
    x, ex, density=True, fill=False, histtype="step", label="empirical pdf"
)
pl.plot(ex, lamb * np.exp(-lamb * ex), label="exponential pdf")

pl.plot(ex, mixture_pdf(ex[:, None]), "--", label="mixture pdf")
pl.errorbar(
    mu_.squeeze(-1),
    mixture_pdf(mu_),
    None,
    s2_.squeeze(-1) ** 0.5,
    "x",
    capsize=2,
    label="mixture modes",
)

pl.legend(loc="upper right")
pl.xlim(xmax=xmax)


# In[ ]:
