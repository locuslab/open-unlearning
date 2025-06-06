<div align="center">

# Reproduce Results

</div>

> [!TIP]
> ​This page is for reproducibility. For results where methods have been tuned for optimal performance, please refer to the [`community/leaderboard`](../community/leaderboard.md).

The scripts below execute standard baseline unlearning experiments on the TOFU and MUSE datasets, evaluated using their corresponding benchmarks. 
```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

## Results



For all the experiments below, we used the following setup

| **Category**            | **Details**                                                                                                                                                                                                                                                                                                             |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Hardware**           | 2 × L40s GPUs (48GB each)                                                                                                                                                                                                                                                                                               |
| **Distributed Computing** | [DeepSpeed ZeRO Stage 3 (Accelerate)](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)                                                                                                                                                                                                                 |
| **Hyperparameters**    | Learning Rate (lr) = 1e-5 <br> preferences=[1.0, 1.0], β = 0.1 (where applicable) <br> Batch size 32 effectively: 8 per device, 4 grad accum steps <br> Number of Epochs = 10 <br> Optimizer: [paged_adamw_32bit](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw#bitsandbytes.optim.PagedAdamW) |


> [!NOTE] 
> 1. The results in the next section display only some important subsets of metrics for each benchmark. For examples of more available evaluation metrics available: see `muse*/*_SUMMARY.json`, `tofu*/evals*/*_SUMMARY.json` files on the [HuggingFace space](https://huggingface.co/datasets/open-unlearning/eval).
> 2. Results may vary even with the same effective hyperparameters when trained with modifications to the distributed training setup, including when training on a single GPU. For example: methods such as SimNPO & RMU can be significantly improved with careful tuning. **Please use the below numbers only for reproducibility purposes**.
> 3. __NPO inconsistency__: for NPO, the MUSE implementation is inconsistent with the [original paper](https://github.com/licong-lin/negative-preference-optimization) as discussed [here](https://github.com/jaechan-repo/muse_bench/issues/2). This inconsistency is carried over into implementations like [SimNPO](https://github.com/OPTML-Group/Unlearn-Simple/issues/5). Here, we use the original NPO implementation with the same loss function expression across datasets.



### TOFU unlearning on the `Llama-2-7b-hf-chat` architecture

<div style="overflow-x: auto; max-width: 100%;"t>
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th style="text-align: center;" colspan="3" halign="left">forget01</th>
      <th style="text-align: center;" colspan="3" halign="left">forget05</th>
      <th style="text-align: center;" colspan="3" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>1.27e-03</td>
      <td>0.63</td>
      <td>0.53</td>
      <td>5.87e-14</td>
      <td>0.63</td>
      <td>0.51</td>
      <td>4.35e-25</td>
      <td>0.63</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>1.0</td>
      <td>0.63</td>
      <td>0.68</td>
      <td>1.0</td>
      <td>0.63</td>
      <td>0.67</td>
      <td>1.0</td>
      <td>0.61</td>
      <td>0.68</td>
    </tr colspan=20>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>1.88e-04</td>
      <td>0.55</td>
      <td>0.36</td>
      <td>1.94e-119</td>
      <td>0.00e+00</td>
      <td>8.82e-96</td>
      <td>1.06e-239</td>
      <td>0.00e+00</td>
      <td>2.21e-32</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>3.02e-03</td>
      <td>0.57</td>
      <td>0.41</td>
      <td>1.94e-119</td>
      <td>0.56</td>
      <td>4.14e-95</td>
      <td>1.80e-229</td>
      <td>0.58</td>
      <td>1.46e-07</td>
    </tr>
    <tr>
      <th>IdkDPO</th>
      <td>0.1</td>
      <td>0.56</td>
      <td>0.67</td>
      <td>4.02e-06</td>
      <td>0.04</td>
      <td>0.67</td>
      <td>5.42e-13</td>
      <td>0.04</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>0.4</td>
      <td>0.58</td>
      <td>0.65</td>
      <td>0.09</td>
      <td>0.53</td>
      <td>0.71</td>
      <td>0.42</td>
      <td>0.54</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>1.27e-03</td>
      <td>0.58</td>
      <td>0.41</td>
      <td>1.06e-106</td>
      <td>0.6</td>
      <td>3.94e-05</td>
      <td>1.47e-198</td>
      <td>0.6</td>
      <td>3.17e-04</td>
    </tr>
    <tr>
      <th>RMU</th>
      <td>0.4</td>
      <td>0.62</td>
      <td>0.64</td>
      <td>9.59e-10</td>
      <td>0.02</td>
      <td>0.81</td>
      <td>6.92e-21</td>
      <td>0.03</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>
</div>


### TOFU unlearning on the `Llama-3.2-1B-Instruct` architecture

<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th style="text-align: center;" colspan="3" halign="left">forget01</th>
      <th style="text-align: center;" colspan="3" halign="left">forget05</th>
      <th style="text-align: center;" colspan="3" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
      <th>forget_quality</th>
      <th>model_utility</th>
      <th>forget_truth_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>0.01</td>
      <td>0.6</td>
      <td>0.47</td>
      <td>1.33e-13</td>
      <td>0.6</td>
      <td>0.47</td>
      <td>1.66e-21</td>
      <td>0.6</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>1.0</td>
      <td>0.60</td>
      <td>0.65</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>0.64</td>
      <td>1.0</td>
      <td>0.59</td>
      <td>0.63</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>0.27</td>
      <td>0.33</td>
      <td>0.59</td>
      <td>1.94e-119</td>
      <td>0</td>
      <td>2.52e-23</td>
      <td>1.06e-239</td>
      <td>0</td>
      <td>2.25e-18</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>0.77</td>
      <td>0.43</td>
      <td>0.57</td>
      <td>1.94e-119</td>
      <td>0.53</td>
      <td>3.87e-34</td>
      <td>1.06e-239</td>
      <td>0.49</td>
      <td>3.53e-27</td>
    </tr>
    <tr>
      <th>IdkDPO</th>
      <td>0.01</td>
      <td>0.51</td>
      <td>0.60</td>
      <td>1.12e-05</td>
      <td>0.07</td>
      <td>0.62</td>
      <td>4.64e-12</td>
      <td>0.23</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>0.92</td>
      <td>0.56</td>
      <td>0.66</td>
      <td>0.14</td>
      <td>0.45</td>
      <td>0.7</td>
      <td>0.02</td>
      <td>0.46</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>0.58</td>
      <td>0.46</td>
      <td>0.55</td>
      <td>5.01e-100</td>
      <td>0.58</td>
      <td>4.19e-03</td>
      <td>2.47e-203</td>
      <td>0.54</td>
      <td>1.07e-05</td>
    </tr>
     <tr>
      <th>RMU</th>
      <td>0.16</td>
      <td>0.55</td>
      <td>0.70</td>
      <td>4.87e-10</td>
      <td>0.58</td>
      <td>0.77</td>
      <td>3.15e-15</td>
      <td>0.59</td>
      <td>0.76</td>
    </tr>
  </tbody>
</table>
</div>


### MUSE unlearning on the benchmark's target models

<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th style="text-align: center;">Method</th>
      <th style="text-align: center;" colspan="4" halign="left">News</th>
      <th style="text-align: center;" colspan="4" halign="left">Books</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_knowmem_ROUGE</th>
      <th>forget_verbmem_ROUGE</th>
      <th>privleak</th>
      <th>retain_knowmem_ROUGE</th>
      <th>forget_knowmem_ROUGE</th>
      <th>forget_verbmem_ROUGE</th>
      <th>privleak</th>
      <th>retain_knowmem_ROUGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>0.64</td>
      <td>0.58</td>
      <td>-99.81</td>
      <td>0.56</td>
      <td>0.47</td>
      <td>1.0</td>
      <td>-57.26</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>0.33</td>
      <td>0.20</td>
      <td>0</td>
      <td>0.56</td>
      <td>0.3</td>
      <td>0.14</td>
      <td>0</td>
      <td>0.69</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
      <th>GradAscent</th>
      <td>0</td>
      <td>0</td>
      <td>52.11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.67</td>
      <td>0</td>
    </tr>
    <tr>
      <th>GradDiff</th>
      <td>0.41</td>
      <td>8.92e-03</td>
      <td>93.23</td>
      <td>0.37</td>
      <td>0.18</td>
      <td>0.16</td>
      <td>-37.79</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>NPO</th>
      <td>0.56</td>
      <td>0.35</td>
      <td>-86.00</td>
      <td>0.51</td>
      <td>0.32</td>
      <td>0.84</td>
      <td>-54.24</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>SimNPO</th>
      <td>0.54</td>
      <td>0.36</td>
      <td>-86.11</td>
      <td>0.51</td>
      <td>0.32</td>
      <td>0.84</td>
      <td>-54.26</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>RMU</th>
      <td>0.48</td>
      <td>0.05</td>
      <td>56.36</td>
      <td>0.51</td>
      <td>0.29</td>
      <td>0.79</td>
      <td>-60.52</td>
      <td>0.48</td>
    </tr>
  </tbody>
</table>
</div>