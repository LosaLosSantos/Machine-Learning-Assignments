[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2BzNhU74)
# WUST Machine Learning - Laboratory # 5
**Term:** Winter 2024/2025

Created by: [Daniel Kucharczyk](mailto:daniel.kucharczyk@pwr.edu.pl)

**Due:** Nov 27, 2024, 22:59 UTC

---

## Objective
Build a Perceptron classifier from scratch to understand fundamental concepts of neural networks and implement basic logical operations (`AND`, `OR` gates).

### Problem
Implement a single-layer Perceptron that can:
- Learn linearly separable patterns
- Successfully classify binary logical operations
- Handle the inherent limitations (e.g., inability to learn `XOR` operation)

## Requirements
1. Code:
   - Implement Perceptron class with combined weights/bias parameters
   - Include activation function, fit and predict methods
   - Use NumPy for efficient array operations
   - Add comprehensive docstrings

2. Testing:
   - Create unit tests for AND and OR operations
   - Include XOR test to demonstrate limitations
   - Use pytest framework

## Deliverables
1. `perceptron.py`:
   - Complete Perceptron implementation
   - Properly documented methods
   - Type hints for function parameters

2. `test_logical_functions.py`:
   - Test suite with pytest fixtures
   - Documented test cases for logical operations
   - Truth table validations

3. Documentation:
   - Class and method docstrings

4. Code Quality: Pre-commit checks

## Evaluation Criteria

- Correctness of implementations (80%)
- Quality and clarity of code (20%)

Good luck, and happy coding!
