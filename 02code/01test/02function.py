import random
def rand_add():
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    answer = int(input(f"What is {a} + {b}? "))
    if answer == a + b:
        print("Correct!")
    else:
        print(f"Incorrect! The correct answer is {a + b}.")
    

rand_add()
    