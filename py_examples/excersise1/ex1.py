"""
Fibonacci sequence
Fn = F_{n-2} + F_{n-1}, F_1 = F_2 = 1

example: Fibonacci sequence up to the 5th element
F1 = 1
F2 = 1
F3 = F2 + F1 = 2
F4 = F3 + F2 = 3
F5 = F4 + F3 = 5
F6 = F5 + F4 = 8
...
...
Usually, F1 is omitted, 
thus the first 5 Fibonacci numbers are [1, 2, 3, 5, 8]
"""

class Fibonacci:
    # Function 1. 
    # Receives an integer as input 
    def __init__(self, n):
        self.n = n + 1
        self.sequence = [0] * (self.n)
        self.sequence[0] = 1
        self.sequence[1] = 1
        if self.n > 1:
            for i in range(2, self.n):
                self.sequence[i] = self.sequence[i-2] + self.sequence[i-1]
          
    # Function 2.
    # Count iterations            
    def __iter__(self):
        self.curr = 0
        return self
    
    # Function 3.
    # Get the next Fibonacci number
    def __next__(self):
        if self.curr >= self.n:
            raise StopIteration    
        result = self.sequence[self.curr]
        self.curr  += 1
        return result 
    
# Use the iterator class for a specific integer n    
fib = Fibonacci(10)
list_of_numbers = [number for number in fib]

# Print the results
print(list_of_numbers[1:]) # omit F1