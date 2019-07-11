# AML_2019 Coursework, Part 1, Group 3
Gradient Descent for SMM284 An Introduction to Machine Learning 

# 1. Egg Holder Function
<p align="center">
    <img src="egg.png" alt="Egg Holder graph" /></p>

<p align="center">
    <img src="egg2.png" alt="Egg Holder graph" /></p>

For our Gradient Descent Project, we have decided to choose the **Egg Holder Function** to see how different gradient Descent works.

Reference: [Egg Holder Function Reference](https://www.sfu.ca/~ssurjano/egg.html)



# 2. Introduction
This project aims to explore the global minimum of egg holder function with different gradient descent methods 

The following variant of gradient descent have been used in our project.

## 2.1 Plain Vanilla
 No.|Eta| Converged Steps | Achieved Coordinates | Loss fn
------------ | ------------ | ----------- |------------- | -------------
1|6.5 | 999 | (200.89,536.18) |-624.78
2|0.3 | 231| (439.48,453.98) |-935.34 
3|0.005 | 999| (419.41,434.65) |-860.87
## 2.2 Momentum 
Reference: https://medium.com/@hengluchang/visualizing-gradient-descent-with-momentum-in-python-7ef904c8a847

Momentum (1964) make use of the moving averages of the gradient instead of just taking one value like in plain vanilla gradient descent. It can accumulate velocity in the direction where the gradient is pointing towards the same direction across iterations. It achieves this by adding a portion of the previous weight update to the current one. 

We first initialize our weights at (400.1,400.1) in the ravine loss surface (egg holder) we've created earlier. Then, we experiment with different learning rate eta1(6.5), eta2(0.3), eta3(0.005) run for 1000 iterations and see how it reach to the global minimum f(x)=-959.64, at x = (512, 404.23). Compares to Plain Vanilla approach (231 steps), it takes 114 less steps to reach the global minima (f(x)=-935.33, x= (439.48,453.98)) with 117 steps under same learning rate eta2 (0.3). This is due to momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, it gains faster convergence and reduced oscillation. Reference: http://ruder.io/optimizing-gradient-descent/index.html#momentum Besides, if the stepsize set to be too large (eta1), although the gradient descend converges the earliest, it cannot reach to a minimum as low as the others. This is because large step size can pass over the true minimum and bounce back to higher point (-888.95). Similarly, if the step size too small (eta3), it requests the same number of step size as PV approach to achieve the global minima (-955.25) which results in a higher computational cost.

 No.|Eta| Converged Steps | Achieved Coordinates | Loss fn
------------ | ------------ | ----------- |------------- | -------------
1|1.5 | 115 | (347.33,499.42) |-888.95
2|0.3 | 117| (439.48,453.98) |-935.34
3|0.005 | 999| (439.1,453.98) |-935.31


## 2.3 Adam 

Adaptive Moment Estimation (Adam) multiply a positive factor to the learning rate and moving averages of the gradient. In addition to storing an exponentially decaying average of past squared gradients vt, Adam also keeps an exponentially decaying average of past gradients mt similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface We compute the decaying averages of past mt and past vt squared gradients. mt and vt are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively, hence the name of the method. 

Based on results of our experiments, the Adam approach has much higher computational cost compare to Momentum. With the same learning rate eta1(6.5) eta2(0.3） eta3（0.005) and same initialised weights at (400.1,400.1), it takes 1894 steps to reach the same global minima(-935.33) whereas momentum only takes 117 steps. This might due to the complication of the Egg holder surface contains multi-minima bottom points.

 No.|Eta| Converged Steps | Achieved Coordinates | Loss fn
------------ | ------------ | ----------- |------------- | -------------
1|10.5 | 216 | (439.48,453.98) |-935.34
2|6.5 | 236| (439.48,453.98)  |-935.34
3|0.5 | 773| (439.48,453.98)  |-935.34
# 3. Conclusion
![3_in_one_gif](ezgif.com-video-to-gif.gif)
![3_in_one_graph](3_in_1_GD.png)

 Type of GD|Eta| Converged Steps | Achieved Coordinates | Loss fn
------------ | ------------ | ----------- |------------- | -------------
Adam|0.3 |499 | (437.99,452.51)  |-934.91
Momentum|0.3 |117| (439.48,453.98)   |-935.34
Plain Vanilla|0.3| 231| (439.48,453.98)  |-935.34

When we tried to plot 3 approaches (PV, Momentu, Adam) together with same learning rate 0.3 and initial point (400.1,400.1), we can conclude that Momentum has the best performance among other 2 methods. Momentum converged in 117 steps to reach the local minimum -935.33 whereas Adam preforms worse than PV which has the highest computational cost.

