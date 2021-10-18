# -*- coding: utf-8 -*-

t = [2, 5, 3, 3, 3, 4, 4, 1, 2, 2, 4, 5, 2, 3, 1]
# Python dictionary method get() returns a value for the given key.
# If key is not available then returns default value None.
hist = {}
for i in t:
    hist[i] = hist.get(i, 0) + 1

# Add r'py\statistics' to the system path
import sys
sys.path.append('py/statistics/think_stats2')
import thinkstats2

hist = thinkstats2.Hist([1, 2, 2, 3, 5])
hist.Freq(4)
hist.Values()

for val in sorted(hist.Values()):
    print(val, hist.Freq(val))

hist.Items()

for value, item in hist.Items():
    print(value, item)


import thinkplot
thinkplot.Hist(hist)
thinkplot.Show(xlabel='value', ylabel='frequency')

sys.path.append('py/statistics/think_stats2/workshop')
import nsfg
preg = nsfg.ReadFemPreg()
live = preg.loc[preg['outcome'] == 1, :]

hist = thinkstats2.Hist(live['birthwgt_lb'], label='birthwgt_lb')
thinkplot.Hist(hist)
thinkplot.Show(xlabel='pounds', ylabel='frequency')

for weeks, freq in hist.Smallest(10):
    print(weeks, freq)

hist.Largest(10)

firsts = live[live['birthord'] == 1]
others = live[live['birthord'] != 1]

first_hist = thinkstats2.Hist(firsts['prglngth'])
other_hist = thinkstats2.Hist(others['prglngth'])

width = .45
thinkplot.PrePlot(2)
thinkplot.Hist(first_hist, align='right', width=width)
thinkplot.Hist(other_hist, align='left', width=width)
thinkplot.Show(xlabel='weeks', ylabel='frequency', xlim=[27, 46])



