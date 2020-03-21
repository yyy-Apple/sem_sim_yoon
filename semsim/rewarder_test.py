from semsim.rewarder import Rewarder

if __name__ == '__main__':
    rewarder = Rewarder(gpu=False)
    reward1 = rewarder("This is just a test.", "This is an interesting test")
    reward2 = rewarder("I want to eat something hot.", "I want to eat hot pot.")
    reward3 = rewarder("Hello", "Hello")
    print(reward1.item())
    print(reward2.item())
    print(reward3.item())
