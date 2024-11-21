[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AUoF7uBb)
# WUST Machine Learning - Laboratory # 3
**Term:** Winter 2024/2025

Created by: [Daniel Kucharczyk](mailto:daniel.kucharczyk@pwr.edu.pl)

---

## Objective
The aim of this homework is to implement and compare several advanced optimization methods for gradient descent. You will build upon the code you developed in the previous lab to implement these new algorithms.

### Problems

1. **Momentum Gradient Descent**
   - Add a momentum term to accumulate past gradients.
   - Update rule:

     $$v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta_t)$$

     $$\theta_{t+1} = \theta_t - \alpha v_t$$

   where:
   - $v_t$ is the velocity vector
   - $\beta$ is the momentum coefficient (typically 0.9)
   - $\alpha$ is the learning rate
   - $\nabla J(\theta_t)$ is the gradient of the cost function


2. **Nesterov Accelerated Gradient (NAG)**
   - This is similar to Momentum, but the gradient is evaluated at an estimated future position.
   - Update rule:

     $$v_t = \beta v_{t-1} + \nabla J(\theta_t - \alpha \beta v_{t-1})$$

     $$\theta_{t+1} = \theta_t - \alpha v_t$$


3. **AdaGrad (Adaptive Gradient)**
   - This method adapts the learning rate for each parameter based on historical gradients.
   - Update rule:

     $$G_t = G_{t-1} + (\nabla J(\theta_t))^2$$

     $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla J(\theta_t)$$

   where:
   - $G_t$ is the sum of squared gradients up to time t
   - $\epsilon$ is a small constant to avoid division by zero
   - $\odot$ denotes element-wise multiplication


4. **RMSprop**
   - This is an adaptation of AdaGrad that uses a moving average of squared gradients.
   - Update rule:

     $$E(g^2)_t = \beta E(g^2)_{t-1} + (1-\beta)(\nabla J(\theta_t))^2$$

     $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E(g^2)_t + \epsilon}} \odot \nabla J(\theta_t)$$

   where:

   - $E[g^2]_t$ is the exponential moving average of squared gradients
   - $\beta$ is the decay rate (typically 0.9)


5. **Adam (Adaptive Moment Estimation)**
   - This method combines ideas from both Momentum and RMSprop.
   - Update rule:

     $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla J(\theta_t)$$

     $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla J(\theta_t))^2$$

     $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

     $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

     $$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

   where:
   - $m_t$ is the first moment estimate (mean of gradients)
   - $v_t$ is the second moment estimate (uncentered variance of gradients)
   - $\beta_1$ and $\beta_2$ are decay rates (typically $0.9$ and $0.999$)
   - $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected moment estimates

## Requirements

1. Reuse and modify your existing code from the previous lab as a starting point.
3. Ensure that your implementations can work with any given loss function and its gradient.
4. Test each method on at least two different optimization problems (e.g., linear regression and logistic regression).
5. Compare the convergence rates and final performance of each method.

## Deliverables

1. Python code implementing all five optimization methods.
2. A brief report (2-3 pages) containing:
   - A description of each implemented method.
   - Plots showing the convergence of the loss function for each method.
   - A comparison of the performance of different methods.
   - A discussion of your observations and conclusions.

## Evaluation Criteria

- Correctness of implementations (50%)
- Quality and clarity of code (20%)
- Thoroughness of testing and comparison (20%)
- Quality of report and insights (10%)

## Submission Guidelines
- Include your report as a jupyter notebook document (`notebooks`).
- Submit all files via the course management system by 1st of November.

Good luck, and happy coding!
