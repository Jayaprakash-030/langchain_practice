from pydantic import BaseModel, EmailStr, Field
import email
from typing import Optional


class Student(BaseModel):
    name: str = "nitish"
    age: Optional[int] = None  # for optional values you should mentioned them as None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing the cgpa of the student")


new_student = {
    "name": "Jayaprakash",
    "age": "32", # it can do type coercing
    "email": "abc@gmail.com",
    "cgpa": 9
}  

student = Student(**new_student) #pydantic class object

student_dict = student.model_dump(mode='dict') #dictionary object

print(student)
print(student_dict)
