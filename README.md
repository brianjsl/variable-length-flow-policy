This repository contains the original implementation for "Learning Long-Context Diffusion Policies via Past-Token Prediction". 

Website: https://long-context-dp.github.io

Paper: https://arxiv.org/abs/2505.09561

We are very grateful for the great work done by Chi et al with the [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) github repository that we deeply inspired our repo from.

# Installation (by EOD) 

# Running the code

The following files may be of use for training:
 - `experiment_configs/`: a folder with all configurations used for experiments
 - `transformer_history.sh`: a shell script accessing some of the common modes for experiments
 - `gather_rollouts.py`: a shell script gathering rollouts for the action predictability analysis in `/rollouts`
 - `rollouts/merge_actions.py`: a script to merge togther several data files gathered for predictability analysis
 - `rollouts_via_policy.py`: a script for measuring the predictability of a rollout data file
 - `transformer_eval.sh`: a script for evaluating checkpoints (with various amounts of test-time consistency)

## Coming Soon (by EOD)

 - Data and cached observation encoder checkpoints
 - Improved documentation

# Citation

If you use the code, remember to cite our paper:

```
@misc{torne2025learninglongcontextdiffusionpolicies,
      title={Learning Long-Context Diffusion Policies via Past-Token Prediction}, 
      author={Marcel Torne and Andy Tang and Yuejiang Liu and Chelsea Finn},
      year={2025},
      eprint={2505.09561},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.09561}, 
}
```



