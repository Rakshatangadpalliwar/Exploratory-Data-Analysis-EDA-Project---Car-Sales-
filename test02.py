
class Inventory:
    products = {}
    def __init__(self):
        pass

    def addition(self):
        self.item_name = input("Enter Item NAME :: ")
        self.quantity = int(input("Enter The Quantity :: "))

        if self.item_name not in Inventory.products.keys():
            self.products[self.item_name] = self.quantity
        else:
            self.products[self.item_name] += self.quantity

    def removing(self):
        self.item_name = input("Enter Item NAME :: ")
        if self.item_name not in Inventory.products.keys():
            print("Item Not Found")
            return
        else:
            self.quantity = int(input("Enter The Quantity :: "))
            self.products[self.item_name] -= self.quantity

    def display(self):
        print(f"{self.item_name}    |    {self.quantity}")

    @classmethod
    def total_stock(cls):
        print(cls.products)


p1 = Inventory()
choice = 1
while choice !=0:
    print("-----------------------------Menu-----------------------------")
    print(f"1 : Add item")
    print(f"2 : Remove item")
    print(f"3 : Display item")
    print(f"4 : TotAL Stock")
    print(f"0 : Exit")

    choice = int(input("Enter Your Choice : "))

    if choice == 1 :
        p1.addition()

    elif choice == 2 :
        p1.removing()

    elif choice == 3 :
        p1.display()

    elif choice == 4:
        p1.total_stock()

    elif choice == 0 :
        print("EXIT")
        print("------------------------------------------------Thank You--------------------------------------")
        break
    else:
        print("Invalid choice !!")
        print("*******************************  Enter Correct Choice  *********************************")

