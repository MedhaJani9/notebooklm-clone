# RAG Evaluation: Vector Search vs Hybrid Search

- Corpus: `eval/docs` (124 chunks, ingested in 3.4s), chunk size/overlap from `nbclone/config.py`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` · Answer LLM: `openai/gpt-oss-20b` · Judge: `openai/gpt-oss-120b` · top-k = 5
- **vector**: cosine similarity over ChromaDB. **hybrid**: vector + BM25 merged with Reciprocal Rank Fusion (k=60).
- *Precision@k* = share of retrieved chunks containing an expected keyword; *Keyword coverage* = share of expected keywords found anywhere in retrieved chunks; *Relevance/Groundedness* = LLM-judge scores (1-5).

## Summary

| Method | Precision@k | Keyword coverage | Top-1 hit | Retrieval ms | LLM s | Relevance | Groundedness |
|---|---|---|---|---|---|---|---|
| vector | 0.62 | 1.00 | 1.00 | 4.0 | 0.6 | 4.88 | 4.25 |
| hybrid | 0.58 | 1.00 | 1.00 | 5.8 | 0.6 | 4.88 | 4.50 |

## Per-question results

| Question | Method | Retrieved | Precision@k | Retrieval ms | LLM s | Relevance | Groundedness | Judge note |
|---|---|---|---|---|---|---|---|---|
| How do I add an item to the end of a list and remove an item by its index? | vector | 5 | 0.40 | 5.3 | 1.0 | 5.0 | 3.0 | append supported; pop not in sources |
| How do I add an item to the end of a list and remove an item by its index? | hybrid | 5 | 0.40 | 8.7 | 0.7 | 5.0 | 3.0 | Append supported; pop not in sources |
| What is a tuple and can it be modified after creation? | vector | 5 | 0.80 | 3.6 | 0.4 | 4.0 | 3.0 | Answer matches question; immutability claim partly supported by source 5. |
| What is a tuple and can it be modified after creation? | hybrid | 5 | 0.80 | 6.8 | 0.4 | 5.0 | 4.0 | Answer matches question, mostly supported by sources |
| How does exception handling work with try, except and finally? | vector | 5 | 0.80 | 8.0 | 0.8 | 5.0 | 5.0 | Answer aligns well with source details |
| How does exception handling work with try, except and finally? | hybrid | 5 | 0.80 | 3.4 | 0.6 | 5.0 | 5.0 | Accurately reflects source content on try/except/finally. |
| What does the enumerate function do? | vector | 5 | 0.40 | 3.4 | 0.5 | 5.0 | 5.0 | Answer matches question and is supported by source [1] |
| What does the enumerate function do? | hybrid | 5 | 0.40 | 7.7 | 0.3 | 5.0 | 5.0 | Answer matches question and is directly supported by source |
| How is a factorial function written using recursion? | vector | 5 | 1.00 | 3.2 | 0.6 | 5.0 | 5.0 | Accurate recursive factorial from source |
| How is a factorial function written using recursion? | hybrid | 5 | 0.80 | 7.3 | 0.4 | 5.0 | 5.0 | Accurate recursive factorial, matches source 1 |
| How does inheritance work and what does super() do? | vector | 5 | 0.80 | 3.2 | 0.8 | 5.0 | 5.0 | Answer matches question and is fully supported by sources |
| How does inheritance work and what does super() do? | hybrid | 5 | 0.80 | 4.4 | 0.4 | 5.0 | 5.0 | Accurate, well‑sourced explanation of inheritance and super() |
| What is a lambda function in Python? | vector | 5 | 0.40 | 3.0 | 0.5 | 5.0 | 5.0 | Answer directly defines lambda and cites source [3] |
| What is a lambda function in Python? | hybrid | 5 | 0.40 | 3.1 | 0.3 | 5.0 | 5.0 | Answer directly defines lambda and cites source |
| What are decorators in Python? | vector | 5 | 0.40 | 2.5 | 0.4 | 5.0 | 3.0 | General definition not directly in sources, but uses decorator examples. |
| What are decorators in Python? | hybrid | 5 | 0.20 | 5.0 | 1.4 | 4.0 | 4.0 | Answer matches question and uses source examples, minor inference. |

## Retrieved chunks and answers

### How do I add an item to the end of a list and remove an item by its index? — *vector*

1. `Python.pdf` p. 18 (score 0.532): 17    CHAPTER 4 – LISTS AND TUPLES   Python lists are containers to store a set of values of any data type.           LIST INDEXING   A list can be indexed just…
2. `Python.pdf` p. 52 (score 0.3739): directly or imported to another file.  THE GLOBAL KEYWORD   ‘global’ keyword is used to modify the variable outside of the current scope.  ENUMERATE FUNCTION IN…
3. `Python.pdf` p. 19 (score 0.3058): 18    a = (1, 7, 2)  • a.count (1): a count (1) will return number of times 1 occurs in a.  • a.index (1) will return the index of first occurrence of 1 in a.…
4. `Python.pdf` p. 28 (score 0.3024): print(item)  else:      print("done") # this is printed when the loop exhausts!  Output:  1  7  8  done      THE BREAK STATEMENT   ‘break’ is used to come out o…
5. `Python.pdf` p. 22 (score 0.2993): 21    If you are a programming beginner without much knowledge of mathematical  operations on sets, you can simply look at sets in python as data types containi…

**Answer:** To add an item to the end of a list, use the `append()` method.  
To remove an item by its index, use the `pop(index)` method.  

Both operations are shown in the list methods section of the source.  
[1]

### How do I add an item to the end of a list and remove an item by its index? — *hybrid*

1. `Python.pdf` p. 18 (score 0.0325): 17    CHAPTER 4 – LISTS AND TUPLES   Python lists are containers to store a set of values of any data type.           LIST INDEXING   A list can be indexed just…
2. `Python.pdf` p. 52 (score 0.0325): directly or imported to another file.  THE GLOBAL KEYWORD   ‘global’ keyword is used to modify the variable outside of the current scope.  ENUMERATE FUNCTION IN…
3. `Python.pdf` p. 29 (score 0.0303): 28    Example:  for i in range (0,80):      print(i)     # this will print 0,1,2 and 3      if i==3          break  THE CONTINUE STATEMENT   ‘continue’ is used …
4. `Python.pdf` p. 28 (score 0.0295): print(item)  else:      print("done") # this is printed when the loop exhausts!  Output:  1  7  8  done      THE BREAK STATEMENT   ‘break’ is used to come out o…
5. `Python.pdf` p. 22 (score 0.0295): 21    If you are a programming beginner without much knowledge of mathematical  operations on sets, you can simply look at sets in python as data types containi…

**Answer:** To add an item to the end of a list, use the `append()` method, which appends the element to the list’s end.  
To remove an item by its index, use the `pop(index)` method, which deletes the element at the specified index and returns it.  

Both operations are shown in the list methods section of the source. [1]

### What is a tuple and can it be modified after creation? — *vector*

1. `Python.pdf` p. 18 (score 0.4677): a = () # empty tuple  a = (1,) # tuple with only one element needs a comma  a = (1,7,2) # tuple with more than one element    TUPLE  METHODS   Consider the foll…
2. `Python.pdf` p. 18 (score 0.4135): 17    CHAPTER 4 – LISTS AND TUPLES   Python lists are containers to store a set of values of any data type.           LIST INDEXING   A list can be indexed just…
3. `Python.pdf` p. 50 (score 0.3911): 49    The syntax of types looks something like this:    from typing import List, Tuple, Dict, Union    # List of integers  numbers: List[int] = [1, 2, 3, 4, 5] …
4. `Python.pdf` p. 41 (score 0.3382): 40    __ INIT__()  CONSTRUCTOR   __init__() is a special method which is first run as soon as the object is created.  __init__() method is also known as constru…
5. `Python.pdf` p. 20 (score 0.3145): 19    CHAPTER 4 - PRACTICE  SET   1. Write a program to store seven fruits in a list entered by the user.  2. Write a program to accept marks of 6 students and …

**Answer:** A tuple is an immutable data type in Python that stores a fixed sequence of values. Once a tuple is created, its contents cannot be changed, added to, or removed from; attempts to modify it will raise an error. This immutability is a defining feature of tuples. [1][2][5]

### What is a tuple and can it be modified after creation? — *hybrid*

1. `Python.pdf` p. 18 (score 0.0315): 17    CHAPTER 4 – LISTS AND TUPLES   Python lists are containers to store a set of values of any data type.           LIST INDEXING   A list can be indexed just…
2. `Python.pdf` p. 18 (score 0.0313): a = () # empty tuple  a = (1,) # tuple with only one element needs a comma  a = (1,7,2) # tuple with more than one element    TUPLE  METHODS   Consider the foll…
3. `Python.pdf` p. 50 (score 0.0306): 49    The syntax of types looks something like this:    from typing import List, Tuple, Dict, Union    # List of integers  numbers: List[int] = [1, 2, 3, 4, 5] …
4. `Python.pdf` p. 20 (score 0.0305): 19    CHAPTER 4 - PRACTICE  SET   1. Write a program to store seven fruits in a list entered by the user.  2. Write a program to accept marks of 6 students and …
5. `Python.pdf` p. 23 (score 0.0292): 22    CHAPTER 5 – PRACTICE SET   1. Write a program to create a dictionary of Hindi words with values as their English  translation. Provide user with an option…

**Answer:** A tuple is an immutable data type in Python, meaning once it is created its contents cannot be changed. This immutability prevents adding, removing, or altering elements after the tuple has been instantiated. [1][2]

### How does exception handling work with try, except and finally? — *vector*

1. `Python.pdf` p. 51 (score 0.6353): print(e)  When the exception is handled, the code flow continues without program interruption.  We can also specify the exception to catch like below:  try:    …
2. `Python.pdf` p. 52 (score 0.6334): 51    try:      # Somecode  except:      # Somecode  else:      # Code         # This is executed only if the try was successful    TRY WITH FINALLY   Python of…
3. `Python.pdf` p. 51 (score 0.4681): 50    dict1 = {'a': 1, 'b': 2}  dict2 = {'b': 3, 'c': 4}  merged = dict1 | dict2  print(merged)  # Output: {'a': 1, 'b': 3, 'c': 4}  You can now use multiple co…
4. `Python.pdf` p. 6 (score 0.4518): Raising Exceptions ................................ ................................ ................................ ........................ 50  try with else…
5. `Python.pdf` p. 29 (score 0.3872): 28    Example:  for i in range (0,80):      print(i)     # this will print 0,1,2 and 3      if i==3          break  THE CONTINUE STATEMENT   ‘continue’ is used …

**Answer:** Exception handling in Python is performed with a `try` block that encloses code that might raise an error, followed by one or more `except` clauses that catch specific exception types (or a generic `except`), and optionally a `finally` clause that runs regardless of whether an exception occurred. The flow is:

1. **`try` block** – code that may raise an exception is executed.  
2. **`except` clauses** – if an exception is raised, Python searches the `except` clauses in order. The first matching clause is executed, and the program continues after the `except` block. If no clause matches, the exception propagates up the call stack.  
3. **`finally` clause** – code in a `finally` block is executed no matter what: whether an exception was raised, caught, or the block completed normally. It is typically used for cleanup actions such as closing files or releasing resources.  

Thus, the `finally` block guarantees that cleanup code runs even if an error occurs or is caught, ensuring that resources are properly released.  

[1][2]

### How does exception handling work with try, except and finally? — *hybrid*

1. `Python.pdf` p. 52 (score 0.0325): 51    try:      # Somecode  except:      # Somecode  else:      # Code         # This is executed only if the try was successful    TRY WITH FINALLY   Python of…
2. `Python.pdf` p. 51 (score 0.0323): print(e)  When the exception is handled, the code flow continues without program interruption.  We can also specify the exception to catch like below:  try:    …
3. `Python.pdf` p. 51 (score 0.032): 50    dict1 = {'a': 1, 'b': 2}  dict2 = {'b': 3, 'c': 4}  merged = dict1 | dict2  print(merged)  # Output: {'a': 1, 'b': 3, 'c': 4}  You can now use multiple co…
4. `Python.pdf` p. 6 (score 0.0312): Raising Exceptions ................................ ................................ ................................ ........................ 50  try with else…
5. `Python.pdf` p. 27 (score 0.0286): 26    CHAPTER 7 – LOOPS IN PYTHON   Sometimes we want to repeat a set of statements in our program. For instance:  Print 1  to 1000.  Loops make it easy for a p…

**Answer:** Exception handling in Python uses a `try` block to enclose code that might raise an error, an `except` clause to catch and process specific or all exceptions, and an optional `finally` clause that is executed regardless of whether an exception occurred or was caught. The flow is:

1. **`try`** – The code inside the `try` block is executed.  
2. **`except`** – If an exception is raised, Python looks for a matching `except` clause. If found, the code in that clause runs; otherwise the exception propagates upward.  
3. **`finally`** – After the `try` (and any matching `except`) finishes, the `finally` block runs whether an exception was raised or not, making it suitable for cleanup actions such as closing files or releasing resources.  

This structure ensures that cleanup code in `finally` is always executed, even if the `try` block succeeds or an exception is re‑raised.  

[1][4]

### What does the enumerate function do? — *vector*

1. `Python.pdf` p. 52 (score 0.4827): directly or imported to another file.  THE GLOBAL KEYWORD   ‘global’ keyword is used to modify the variable outside of the current scope.  ENUMERATE FUNCTION IN…
2. `Python.pdf` p. 6 (score 0.4711): enumerate function in python ................................................................ ................................ ........ 51  List comprehensions …
3. `Python.pdf` p. 31 (score 0.4655): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…
4. `Python.pdf` p. 32 (score 0.4284): 31    def greet(name):      gr = "hello"+ name       return gr                        a = greet ("harry")  # a will now contain "hello harry"     DEFAULT PAR AM…
5. `Python.pdf` p. 32 (score 0.42): factorial(n) = n x factorial (n-1)  This function can be defined as follows:  def factorial(n)      if i == 0 or i==1: # base condition which doesn’t call the f…

**Answer:** The `enumerate` function in Python adds a counter (index) to an iterable and returns an iterator that yields pairs of the form `(index, item)`, allowing you to loop over both the index and the element simultaneously. [1][2]

### What does the enumerate function do? — *hybrid*

1. `Python.pdf` p. 52 (score 0.0328): directly or imported to another file.  THE GLOBAL KEYWORD   ‘global’ keyword is used to modify the variable outside of the current scope.  ENUMERATE FUNCTION IN…
2. `Python.pdf` p. 6 (score 0.0323): enumerate function in python ................................................................ ................................ ........ 51  List comprehensions …
3. `Python.pdf` p. 31 (score 0.0286): 30    CHAPTER 8 – FUNCT IONS & RECURSIONS   A function is a group of statements performing a specific task.  When a program gets bigger in size and its complexi…
4. `Python.pdf` p. 31 (score 0.0284): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…
5. `Python.pdf` p. 29 (score 0.0272): 28    Example:  for i in range (0,80):      print(i)     # this will print 0,1,2 and 3      if i==3          break  THE CONTINUE STATEMENT   ‘continue’ is used …

**Answer:** The `enumerate` function adds a counter (index) to each item of an iterable and returns an iterator of `(index, item)` pairs, allowing you to loop over both the index and the value simultaneously. [1]

### How is a factorial function written using recursion? — *vector*

1. `Python.pdf` p. 32 (score 0.7184): factorial(n) = n x factorial (n-1)  This function can be defined as follows:  def factorial(n)      if i == 0 or i==1: # base condition which doesn’t call the f…
2. `Python.pdf` p. 32 (score 0.6429): 31    def greet(name):      gr = "hello"+ name       return gr                        a = greet ("harry")  # a will now contain "hello harry"     DEFAULT PAR AM…
3. `Python.pdf` p. 33 (score 0.5698): 32    This works as follows:                                                              The programmer needs to be extremely careful while working with recurs…
4. `Python.pdf` p. 31 (score 0.49): 30    CHAPTER 8 – FUNCT IONS & RECURSIONS   A function is a group of statements performing a specific task.  When a program gets bigger in size and its complexi…
5. `Python.pdf` p. 4 (score 0.4861): Chapter 8 – Functions & Recursions ................................ ................................................................ .. 30  Example and syntax o…

**Answer:** A factorial function written with recursion can be defined as follows:

```python
def factorial(n):
    if n == 0 or n == 1:          # base case – stops the recursion
        return 1
    else:
        return n * factorial(n-1) # recursive call
```

This definition follows the recursive formula factorial(n) = n × factorial(n‑1) and stops when n reaches 0 or 1, returning 1. [1]

### How is a factorial function written using recursion? — *hybrid*

1. `Python.pdf` p. 32 (score 0.0325): factorial(n) = n x factorial (n-1)  This function can be defined as follows:  def factorial(n)      if i == 0 or i==1: # base condition which doesn’t call the f…
2. `Python.pdf` p. 32 (score 0.0325): 31    def greet(name):      gr = "hello"+ name       return gr                        a = greet ("harry")  # a will now contain "hello harry"     DEFAULT PAR AM…
3. `Python.pdf` p. 33 (score 0.0317): 32    This works as follows:                                                              The programmer needs to be extremely careful while working with recurs…
4. `Python.pdf` p. 34 (score 0.0299): 33    CHAPTER 8 – PRACTICE SET   1. Write a program using functions to find greatest of three numbers.  2. Write a python program using function to convert Cels…
5. `Python.pdf` p. 30 (score 0.0296): 29    CHAPTER 7 – PRACTICE SET   1. Write a program to print multiplication table of a given number using for loop.  2. Write a program to greet all the person …

**Answer:** A factorial function written using recursion calls itself with a smaller argument until it reaches the base case (0 or 1). In Python it can be written as:

```python
def factorial(n):
    if n == 0 or n == 1:          # base condition
        return 1
    else:
        return n * factorial(n-1) # recursive call
```

This definition follows the mathematical recurrence factorial(n) = n × factorial(n‑1) and stops when n is 0 or 1. [1][2]

### How does inheritance work and what does super() do? — *vector*

1. `Python.pdf` p. 44 (score 0.6429): 43    MULTI PLE INHERITANCE   Multiple Inheritance occurs when the child class inherits from more than one parent  classes.     MULTI LEVEL INHERITANCE   When a…
2. `Python.pdf` p. 5 (score 0.576): Multiple Inheritance ................................ ................................ ................................ ...................... 43  Multilevel In…
3. `Python.pdf` p. 43 (score 0.4892): 42    CHA PTER 11 - INHERITANCE  & MORE ON OOPS   Inheritance is a way of creating a new class from an existing class.  Syntax:  class Employee:  # Base class  …
4. `Python.pdf` p. 5 (score 0.379): __init__() constructor ................................ ................................ ................................ ..................... 40  Chapter 10 –…
5. `Python.pdf` p. 41 (score 0.362): 40    __ INIT__()  CONSTRUCTOR   __init__() is a special method which is first run as soon as the object is created.  __init__() method is also known as constru…

**Answer:** Inheritance in Python allows a new class (the child or derived class) to acquire the attributes and methods of an existing class (the parent or base class). The child can use, override, or extend the inherited members, enabling code reuse and polymorphism. Types of inheritance include single, multiple, and multilevel inheritance, where a child may inherit from one, more than one, or from a chain of classes respectively【3】【1】.  

The `super()` function is used inside a derived class to access methods of its superclass. It typically calls the superclass’s constructor (`super().__init__()`) or other methods, allowing the derived class to extend or modify base-class behavior while still invoking the original implementation【1】.

### How does inheritance work and what does super() do? — *hybrid*

1. `Python.pdf` p. 44 (score 0.0328): 43    MULTI PLE INHERITANCE   Multiple Inheritance occurs when the child class inherits from more than one parent  classes.     MULTI LEVEL INHERITANCE   When a…
2. `Python.pdf` p. 5 (score 0.0323): Multiple Inheritance ................................ ................................ ................................ ...................... 43  Multilevel In…
3. `Python.pdf` p. 43 (score 0.0317): 42    CHA PTER 11 - INHERITANCE  & MORE ON OOPS   Inheritance is a way of creating a new class from an existing class.  Syntax:  class Employee:  # Base class  …
4. `Python.pdf` p. 5 (score 0.0306): __init__() constructor ................................ ................................ ................................ ..................... 40  Chapter 10 –…
5. `Python.pdf` p. 40 (score 0.0287): from an object.  harry.getSalary() # here self is harry   # equivalent to Employee.getSalary(harry)  The function getSalary() is defined as:  class Employee:   …

**Answer:** Inheritance in Python allows a new class (the *derived* or *child* class) to reuse, extend, or modify the attributes and methods of an existing class (the *base* or *parent* class). The child class can inherit from one parent (single inheritance), more than one parent (multiple inheritance), or from a chain of parents (multilevel inheritance) [3][1].  

When a child class inherits from a parent, it automatically gains access to the parent’s methods and attributes, and it can override or add its own implementations.  

The `super()` function is used inside a derived class to call a method from its immediate parent class. It is commonly used to invoke the parent’s constructor (`super().__init__()`) so that the base class is properly initialized before the child class adds its own initialization logic. `super()` provides a way to access the superclass’s methods without explicitly naming the parent class, which is especially useful in multiple inheritance scenarios [1].

### What is a lambda function in Python? — *vector*

1. `Python.pdf` p. 55 (score 0.627): Filter creates a list of items for which the function returns true.  list(filter(function))                # the function can be lambda function  Reduce applies…
2. `Python.pdf` p. 31 (score 0.5927): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…
3. `Python.pdf` p. 54 (score 0.5395): containing the output of ‘pip freeze’.  We can distribute this file to other users, and they can recreate the same environment  using:  pip install –r requireme…
4. `Python.pdf` p. 31 (score 0.53): 30    CHAPTER 8 – FUNCT IONS & RECURSIONS   A function is a group of statements performing a specific task.  When a program gets bigger in size and its complexi…
5. `Python.pdf` p. 32 (score 0.4999): 31    def greet(name):      gr = "hello"+ name       return gr                        a = greet ("harry")  # a will now contain "hello harry"     DEFAULT PAR AM…

**Answer:** A lambda function in Python is an anonymous, one‑line function created using the `lambda` keyword. It follows the syntax `lambda arguments: expression`, where the expression is evaluated and returned when the lambda is called. For example, `square = lambda x: x*x` defines a lambda that returns the square of its argument. [3]

### What is a lambda function in Python? — *hybrid*

1. `Python.pdf` p. 55 (score 0.0325): Filter creates a list of items for which the function returns true.  list(filter(function))                # the function can be lambda function  Reduce applies…
2. `Python.pdf` p. 54 (score 0.0323): containing the output of ‘pip freeze’.  We can distribute this file to other users, and they can recreate the same environment  using:  pip install –r requireme…
3. `Python.pdf` p. 31 (score 0.0312): 30    CHAPTER 8 – FUNCT IONS & RECURSIONS   A function is a group of statements performing a specific task.  When a program gets bigger in size and its complexi…
4. `Python.pdf` p. 32 (score 0.0299): 31    def greet(name):      gr = "hello"+ name       return gr                        a = greet ("harry")  # a will now contain "hello harry"     DEFAULT PAR AM…
5. `Python.pdf` p. 31 (score 0.0298): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…

**Answer:** A lambda function in Python is an anonymous, inline function created using the `lambda` keyword. It follows the syntax `lambda arguments: expression`, where the expression is evaluated and returned when the lambda is called. Lambda functions can be used wherever a normal function is expected, such as in `filter`, `map`, or `reduce` calls, and they are often defined as one‑line expressions like `square = lambda x: x*x` or `sum = lambda a,b,c: a+b+c`【2】.

### What are decorators in Python? — *vector*

1. `Python.pdf` p. 45 (score 0.6191): 44    CLASS METHOD   A class method is a method which is bound to the class and not the object of the class.  @classmethod decorator is used to create a class m…
2. `Python.pdf` p. 41 (score 0.5337): 40    __ INIT__()  CONSTRUCTOR   __init__() is a special method which is first run as soon as the object is created.  __init__() method is also known as constru…
3. `Python.pdf` p. 5 (score 0.4612): Newly added features in python ................................ ................................................................ ..... 48  Walrus Operator .....…
4. `Python.pdf` p. 40 (score 0.4567): from an object.  harry.getSalary() # here self is harry   # equivalent to Employee.getSalary(harry)  The function getSalary() is defined as:  class Employee:   …
5. `Python.pdf` p. 31 (score 0.4512): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…

**Answer:** Decorators in Python are special functions or classes that modify the behavior of other functions, methods, or classes. They are applied using the `@decorator_name` syntax placed above the definition of the target function or method, allowing you to wrap or extend its functionality without changing its code directly. This mechanism is commonly used for tasks such as logging, access control, caching, or converting instance methods into class or static methods. [1]

### What are decorators in Python? — *hybrid*

1. `Python.pdf` p. 45 (score 0.0309): 44    CLASS METHOD   A class method is a method which is bound to the class and not the object of the class.  @classmethod decorator is used to create a class m…
2. `Python.pdf` p. 45 (score 0.0308): @name.setter  def name (self,value):      self.ename = value  OPER ATOR OVERLOADI NG IN PYTHON   Operators in Python can be overloaded using dunder methods.  Th…
3. `Python.pdf` p. 31 (score 0.0295): The part containing the exact set of instructions which are executed during the function  call.  Quick Quiz:  Write a program to greet a user with “Good day” us…
4. `Python.pdf` p. 8 (score 0.0292): Some examples of built in modules are os, random etc.  Some examples of external modules are tensorflow, flask etc.  USING PYTHON AS A CALCULATOR   We can use p…
5. `Python.pdf` p. 8 (score 0.0278): 7    CHAPTER 1 – MODULES, COMMENTS & PIP   Let’s write our very first python program. Create a file called hello.py and paste the  below code in it.  print("hel…

**Answer:** Decorators in Python are special functions or methods that modify the behavior of other functions, methods, or classes. They are applied using the `@decorator_name` syntax placed above the definition of the target function or method. Common built‑in decorators include `@classmethod`, `@property`, and the setter form `@name.setter`, which change how a method is bound or accessed within a class [1].

## Conclusions

**Headline numbers (8 questions, top-k = 5, 124 chunks from a 1-PDF corpus)**

| | Vector | Hybrid (BM25 + vector, RRF) |
|---|---|---|
| Precision@5 | **0.62** | 0.58 |
| Keyword coverage / Top-1 hit | 1.00 / 1.00 | 1.00 / 1.00 |
| Retrieval latency | **4.0 ms** | 5.8 ms |
| Judge relevance | 4.88 | 4.88 |
| Judge groundedness | 4.25 | **4.50** |

**1. Retrieval quality is essentially tied on this corpus.** Both methods put a correct chunk at rank 1 for every
question and both retrieved every expected keyword somewhere in the top 5. Vector search has a slightly higher
Precision@5 (0.62 vs 0.58); the gap comes from a single question ("decorators", 0.40 vs 0.20) and one rank-5 swap
("factorial"). With 8 questions, a 0.04 difference is one or two chunks and is not statistically meaningful.

**2. Hybrid produced slightly better-grounded answers.** Groundedness rose from 4.25 to 4.50. Looking at the full
retrieved chunks explains the two questions where it changed:
- *Tuple* (3 → 4): vector search put an off-topic `__init__()` constructor chunk (p. 41) in its top 5, a semantic false
  positive ("modified after **creation**" embeds close to "run as soon as the object is **created**"). Hybrid dropped it
  and moved the p. 18 chunk that states that tuples are *immutable* from rank 2 to rank 1, because BM25 rewards the literal
  term "tuple".
- *Decorators* (3 → 4): hybrid's Precision@5 *fell* (0.40 → 0.20), but partly because of the metric. Hybrid's rank-2 chunk
  is a real decorator example (`@name.setter`, p. 45) that does not contain the word "decorator", so keyword precision
  does not count it. Hybrid also pulled in two off-topic "modules" chunks (p. 8) through weak keyword overlap, and lost
  vector's `@staticmethod` chunk (p. 40).

Chunk-level keyword precision and answer-level usefulness are different metrics. The LLM ignores noisy chunks
as long as the key evidence is present and ranked high.

**3. A measurement artifact, not a retrieval failure.** On the list question both methods scored groundedness 3 with
the judge note "pop not in sources". In fact, both retrieved the correct p. 18 chunk at rank 1, and it contains both
`append()` (char 446) and `pop()` (char 540). The judge prompt truncates each chunk to its first 500 characters, so
the judge never saw `pop()`. The generator, which receives full chunks, answered correctly. This means the groundedness
scores are slightly pessimistic for *both* methods, and it shows why LLM-judge setups need to be checked against the
raw retrieved text.

**4. Latency is not a deciding factor.** Hybrid adds ~2 ms for the BM25 pass and the RRF merge. End-to-end
time is dominated by the LLM call (~0.6 s on Groq), so retrieval is under 1% of response time for either method.
BM25 also costs memory (an in-memory index per notebook) and must be rebuilt after a restart, which is cheap at this
scale (hundreds of chunks) but would matter at hundreds of thousands.

**5. When each method wins (trade-offs).**
- *Vector* handles paraphrases and conceptual questions ("how do I make a value unchangeable?") where the query
  shares no words with the source.
- *Hybrid* is more robust for exact terms: function names (`enumerate`, `super()`), error names, acronyms, code
  identifiers and rare words that a small 384-dim embedding model (MiniLM) represents weakly. Course notes, slides
  and technical PDFs, which is what NotebookLM users upload, are full of these.
- RRF fuses by rank rather than raw score, so it needs no tuning between BM25 and cosine score scales.

**Decision.** **Hybrid is the default** in the app because it gave equal relevance, better groundedness and better
robustness to exact-term queries, for a negligible latency cost. Vector-only stays selectable in the chat UI.

**Limitations of this evaluation.** A small question set (8) over a single document; keyword-based precision is a
proxy for relevance (it misses relevant chunks that don't use the exact keyword, as with `@name.setter`); the judge is
an LLM (`gpt-oss-120b`), which can be lenient, and it only saw the first 500 characters of each chunk (see point 3). Next steps would be a larger
labelled set, a multi-document notebook (where BM25 helps disambiguate between sources) and a cross-encoder reranker
evaluated on top of both methods.
