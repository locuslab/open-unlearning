<div align="center">

# Leaderboard  

</div>  

We encourage the community to develop new methods, optimize them for specific benchmarks, and compare results with existing approaches.  

To implement a new method, refer to our [contributing guide](../docs/contributing.md).  

> [!NOTE]
> The [results.md](../docs/results.md) file is maintained for reproducibility purposes. However, we encourage contributors to update the leaderboard table instead of the reproducibility table. We will continue refining and tuning baseline methods to keep the leaderboard up to date.


### TOFU unlearning on the `Llama-2-7b-hf-chat` architecture

<div style="overflow-x: auto; max-width: 100%;">
<table class="dataframe">
  <thead>
    <tr>
      <th>Method</th>
      <th style="text-align: center;" colspan="2" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>4.35e-25</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>1.0</td>
      <td>0.61</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
    </tr>
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
      <th style="text-align: center;" colspan="2" halign="left">forget10</th>
    </tr>
    <tr>
      <th></th>
      <th>forget_quality</th>
      <th>model_utility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Finetuned</th>
      <td>3.91e-22</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>Retain</th>
      <td>1.0</td>
      <td>0.59</td>
    </tr>
    <tr>
      <td colspan="20"> </td>
    </tr>
    <tr>
    </tr>
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
      <td>-57.34</td>
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
  </tbody>
</table>
</div>
