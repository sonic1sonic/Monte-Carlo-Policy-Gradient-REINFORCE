# Monte-Carlo Policy Gradient (REINFORCE)

Solve the CartPole-v0 with Monte-Carlo Policy Gradient (REINFORCE)!

## How does it work

Take a look at the boxed pseudocode below.

![pseudocode](demo/pseudocode.png)

![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;G_{t}) : return (cumulative discounted **reward**) following time T

![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;\pi(a|s)) : **probability** of taking action ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;a) in state ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;s)

My interpretation of this method is that the actions selected more frequently are the more beneficial choices, thus we try to repeat these actions if similar states are visited.

## Result

![](demo/training_episode_batch_video.gif)

![](demo/learning_performance.png)

## Note

The box pseudocode is from [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/the-book-2nd.html) by [Richard S. Sutton](http://incompleteideas.net/sutton/index.html) and [Andrew G. Barto](http://www-anw.cs.umass.edu/%7Ebarto/).

In my code, the loop of updating weights iterates from ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;t=T-1) to ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;t=0), and the cumulative discounted reward is computed by ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;G_{t}&space;=&space;R_{t&plus;1}&space;&plus;&space;\gamma&space;G_{t&plus;1}).

Besides, the ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;\gamma^{t}) term is ommited in ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;\theta\leftarrow&space;\theta&plus;\alpha\gamma^{t}G\nabla_{\theta}log_{\pi}(A_{t}|S_{t},&space;\theta)).

Since the optimizer will minimize the loss, we will need to multiply the product of ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;G) and ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;log_{\pi}(A_{t}|S_{t},&space;\theta)) by ![](https://latex.codecogs.com/png.latex?\inline&space;\large&space;-1).
