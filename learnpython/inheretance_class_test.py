
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)


class Student(Person):
  # def __init__(self, fname, lname):
  #   super().__init__(fname, lname)# this copies init function

  def test_new_function(self):
      print('test new function')

  def printname(self):
      print(self.lastname, self.firstname)


x = Person("John", "Doe")
x.printname()

y = Student("Mike", "Olsen")
y.test_new_function()
y.printname()


