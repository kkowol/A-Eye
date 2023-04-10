import matplotlib.pyplot as plt
# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')
import csv
import os
from utils.tools import get_folder_name

def pedal_tracking(throttle_sem, throttle_safety, brake_sem, brake_safety, timer):
    """
    saves pedal tracking in a csv file
    """
    path = get_folder_name()
    fields = ["timer", "throttle_sem", "brake_sem", "throttle_safety", "brake_safety"]
    throttle_sem = round((throttle_sem+0.049480200779342)/1.063121089866,2)
    brake_sem=round((brake_sem+0.049480200779342)/1.063121089866,2)
    throttle_safety=round((throttle_safety+0.049480200779342)/1.063121089866,2)
    brake_safety=round((brake_safety+0.049480200779342)/1.063121089866,2)

    row=[timer, throttle_sem, brake_sem, throttle_safety, brake_safety]
    with open(os.path.join(path, '00_log', 'pedal_tracking.csv'), 'a') as file:
        csvwriter = csv.writer(file)
        if not os.path.exists(os.path.join(path, '00_log', 'pedal_tracking.csv')) or os.stat(os.path.join(path, '00_log', 'pedal_tracking.csv')).st_size == 0:
            csvwriter.writerow(fields)
        csvwriter.writerow(row)


def live_plotter(x_vec, y1_data, line1, y2_data, line2, y3_data, line3, pause_time=0.01):
    """
    source: https://github.com/makerportal/pylive
    """
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        
        fig = plt.figure(figsize=(13,6))
        
        ax1 = fig.add_subplot(211, ylim=(-0.2,1.2), xlim=(0,1), ylabel='deflection [%]', title='driving behavior tracking')
        ax2 = fig.add_subplot(212, ylim=(-70,70), xlim=(0,1), ylabel='steering angle [°]')
        
        # ax2 = fig.add_subplot(212)
        # create a variable for the line so we can later update it
        line1, = ax1.plot(x_vec,y1_data,'-o', alpha=0.8, label='throttle')
        line2, = ax1.plot(x_vec,y2_data,'-o',alpha=0.8, label='brake')
        line3, = ax2.plot(x_vec,y3_data,'-o', alpha=0.8, label='steering wheel')     
        
        plt.xlabel('time [x 10s]')
        ax1.legend(loc="upper left") #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
        ax2.legend(loc="upper left")
        
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data</em>
    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    line3.set_ydata(y3_data)
    # adjust limits if new data goes beyond bounds</em>
    # if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
    #     plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above</em>
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration</em>
    return line1, line2, line3