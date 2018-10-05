from bs4 import BeautifulSoup

with open("1.html") as fp:
    soup = BeautifulSoup(fp)

#print(soup.prettify())
print(soup.p)