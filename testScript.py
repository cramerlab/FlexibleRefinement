from matplotlib import pyplot, lines
import numpy

x = numpy.linspace(0,10,100)
y = numpy.sin(x)*(1+x)

fig = pyplot.figure()
ax = pyplot.subplot(111)
ax.plot(x,y,label='a')

# new clear axis overlay with 0-1 limits
#ax2 = pyplot.axes([0,0,1,1], facecolor=(1,1,1,0))

x,y = numpy.array([[0.05, 0.1, 0.9], [0.05, 0.5, 0.9]])
line = lines.Line2D([0,0.5], [-6,-7.1], lw=5., color='r')
ax.add_line(line)

pyplot.show()