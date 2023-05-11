#Regression Python study
#bmi as diabetes predictor
#raw data obtained from kaggle https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
#most functions are hand written. numpy sparsely used

#diabetes prediction
#diabetes is response. 1 or 0 1 for diabetes

#variable column numbers:
#0 = gender                          qual       0
#1 = age                             quant      1
#2 = hypertension (y/n)              quant      2
#3 = heartdisease (y/n)              quant      3
#4 = smoker status                   qual       4
#5 = bmi                             quant      5 
#6 = hbA1c level                     quant      6
#7 = blood glucose levels            quant      7
#8 = diabetes (y/n)                  quant      8

#type one or 2 is not specified and must be noted when drawing conclusions 
#on results


    
#what variables may have coorelation to diabetes and may contribute to
#the existance of a linear model

#most is not hardcoded but some was



#normal imports + copy package. i was having issues duplicating a list and
#had to use this package to solve
import csv
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

#function for reading in data. operates independent of class
def dataRead(x):
    newdat = []
    try:
        with open(x, 'r') as datcsv:
            datread = csv.reader(datcsv)
            header = next(datread)
            newdat.append(header)
            try:
                for row in datread:
                    for entry in row:
                        try:
                            row[row.index(entry)] = format(float(entry),'.2f')
                        except:
                            pass
                    newdat.append(row)
                return(newdat)

            except ValueError: 
                newdat = None
                print(f"Unable to open and read '{x}'.")
                return(newdat)

    except FileNotFoundError:
        newdat = None
        print(f"Unable to open and read '{x}'.")
        return(newdat)

##hardcoded independent function to produce graphical data for
##estimated regression line for bmi as a predictor for diabetes
def histbmi(data):
    data = dataManip(data)
    x = data.x
    
    diabetes = []
    bmi = []
    for i in x:
        for q in i:
            if i.index(q) == 5:
                bmi.append(float(q))
            if i.index(q) == 8:
                diabetes.append(float(q))
    xbar = data.xmeans()[3]
    ybar = data.xmeans()[6]

    i = 0
    sxy=0
    sxx=0
    while i < len(diabetes):
        sxy += (bmi[i] - xbar) * (diabetes[i] - ybar)
        sxx += (bmi[i]- xbar) **2
        i += 1
    b1 = sxy / sxx
    b0 = ybar- (b1 * xbar)


    x = np.linspace(0,100,100)
    y = b0+b1*x
    plt.plot(x, y, '-r', label='bmi x diabetes')
    plt.title('Estimated Regression Line for BMI x Diabetes')
    plt.xlabel('bmi', color='#1C2833')
    plt.ylabel('diabetes', color='#1C2833')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    plt.grid()
    plt.show()



    return (None)

#class contains methods that calculate data parameters and will print an 
#entire formatted dataset if printed

#calculated statistics and a header of data
#are printed through the dataDisplay class

class dataManip:

    def __init__(self, x):
        c = dataRead(x)
        self.xfull = c
        self.xhead = c.pop(0)
        self.x = c
    
    
    def xn (self):
        return(len(self.x))
    
    def xsplit(self):
        x = self.x
        xnum = []
        xcat = []
        for i in x:
            xcatsplice=[]
            xnumsplice=[]
            for q in i:
                try:
                    q = float(q)
                except:
                    pass
                if type(q) == str:
                    xcatsplice.append(q)
                else:
                    xnumsplice.append(q)
            xcat.append(xcatsplice)
            xnum.append(xnumsplice)
            
        return(xnum,xcat)
                    
                
    
    
    #outputs list of means for the 7 numeric variables
    def xmeans (self):
        xnum = self.xsplit()[0]
        means = [0] * 7
        for i in xnum:
            for z in range(0,len(i)):
                means[z] = i[z] + means[z]
        
        for i in range(0,len(means)):
            means[i] = means[i] / self.xn()

        return(means)
    
    #outputs a list of counts for categorical variables
    def xcounts(self):
        xcat = self.xsplit()[1]
        
        xcatuniq = []
        for i in xcat:
            for q in i:
                xcatuniq.append(q)
        xcounts = set(xcatuniq)
        
        a = np.array(xcat)
        unique, counts = np.unique(a, return_counts=True)
        xcounts = dict(zip(unique, counts))
        
        return(xcounts)
        
                
    def xsd (self):
        x = self.xsplit()[0]
        xbar = self.xmeans()
        xsds = [0] * 7
        for i in x:
            for q in range(0,len(xbar)):
                xsds[q] = ((i[q] - xbar[q])**2) + xsds[q]
        for i in range(0,len(xsds)):
            xsds[i] = math.sqrt((xsds[i]) / (self.xn() - 1))
        return(xsds)
    
    def xmax (self):
        x = self.xsplit()[0]
        xmaxs = [0] * 7
        xmaxs0 = []
        xmaxs1 = []
        xmaxs2 = []
        xmaxs3 = []
        xmaxs4 = []
        xmaxs5 = []
        xmaxs6 = []
        for i in x:
            for q in i:
                if i.index(q) == 0:
                    xmaxs0.append(q)
                if i.index(q) == 1:
                    xmaxs1.append(q)
                if i.index(q) == 2:
                    xmaxs2.append(q)
                if i.index(q) == 3:
                    xmaxs3.append(q)
                if i.index(q) == 4:
                    xmaxs4.append(q)
                if i.index(q) == 5:
                    xmaxs5.append(q)
                if i.index(q) == 6:
                    xmaxs6.append(q)
                    
        xmaxs[0]=max(xmaxs0)
        xmaxs[1]=max(xmaxs1)
        xmaxs[2]=max(xmaxs2)
        xmaxs[3]=max(xmaxs3)
        xmaxs[4]=max(xmaxs4)
        xmaxs[5]=max(xmaxs5)
        xmaxs[6]=max(xmaxs6)

        return(xmaxs)
        
    

    #formatted strong for printing data table
    def __str__(self):
        
        x = self.xfull
        #first loop finds correct spacing between data obs 
        #using the largest entry and some integer determind by trial error
        maxval = 0
        for i in x:
            for q in i:
                newmax = len((max((str(q)))))
                if (newmax > maxval):
                    maxval = newmax
        a =""
        maxspaces = 14 + maxval        
        
        #looping makes string formatted list of all entries
        #distance is proportional to lenght of observation string
        for i in x:
            iterat = 0
            for q in i:
                sq =str(q)
                a = a + sq
                spacinglooper = maxspaces - len(sq)
                for g in range(0,spacinglooper):
                    a = a + " "
                iterat += 1
            a = a + "\n"
        return(str())

#class contains methods for formatted printings of results of data
class dataDisplay:
    def __init__(self, x):
        self.x = dataManip(x)
            
    
    #rudimentary summarystats
    def __str__(self):
        a ="\nSummary Statistics\n"
        a= a+ "_____________________\n"
        a = a + "\n"
        headerfull = self.x.xhead
        header = copy.deepcopy(headerfull)
        data = self.x.x
        means = self.x.xmeans()
        sds = self.x.xsd()
        maxs = self.x.xmax()
        counts = self.x.xcounts()
                
        header.pop(0)        
        header.pop(3)
        
        a = a+ "Variables\n"
        a =a+ "__________\n\n"
        a = a+"\n"
        for i in headerfull:
            a = a+ f"#{headerfull.index(i)} - {i}"
            a = a+"\n"
        a = a+"\n\n"
        
        
        a = a+"Means\n"
        a =a+ "_______\n\n"
        i = 0
        for i in range(0,len(means)):
            a = a+f"{header[i]}:\n {means[i]:.2f}"
            a = a+"\n"
        a = a+"\n\n"
        
        
        a = a+"Maxes\n"
        a =a+ "_______\n\n"
        i = 0
        for i in range(0,len(maxs)):
            a = a+f"{header[i]}:\n {maxs[i]:.2f}"
            a = a+"\n"
        a = a+"\n\n"
        
        
        a = a+"Standard Deviations\n"
        a =a+ "___________________\n\n"
        i = 0
        for i in range(0,len(sds)):
            a = a+f"{header[i]}:\n {sds[i]:.2f}"
            a = a+"\n"
        a = a+"\n\n"
        
        
        a= a+ "Occurences of Gender:\n" 
        a= a+ "_____________________\n\n"
        i = 0
        for key in counts:
            a=a + "%s:\n     %s\n" % (key,counts[key])
            i +=1
            if i ==2:           
                a=a+"\n \n" 
                a=a+"Occurences of Smoking Status:\n" 
                a= a+ "___________________________\n\n"
        
        a = a+"\n\n"
        a= a+ "First 10 observations:\n" 
        a= a+ "_______________________\n\n"
        a = a+"\n\n"
        
        
        
    
        x = data.insert(0,headerfull)
        x = data[:11]
        
        maxval = 0
        for i in x:
            for q in i:
                newmax = len((max((str(q)))))
                if (newmax > maxval):
                    maxval = newmax
        maxspaces = 20 + maxval        
        
        #looping makes string formatted list of all entries
        #distance is proportional to lenght of observation string
        
        for i in x:
            iterat = 0
            for q in i:
                sq =str(q)
                a = a + sq
                spacinglooper = maxspaces - len(sq)
                
                for g in range(0,spacinglooper):
                    a = a + " "
                iterat += 1
            a = a + "\n"
            if x.index(i) == 0:
                a = a + "\n"
        
        return(a)






#main
data = ('diabetes.csv')
print(dataDisplay(data))
histbmi(data)
