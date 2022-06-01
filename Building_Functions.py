import numpy as np
import ReadIM
import matplotlib.pyplot as plt
import os
import sys
import easygui
import pandas as pd
from easygui import *

# def data_pull(file_loc,n_contours,ymin_pixel,ymax_pixel,xmin_pixel,xmax_pixel,xrat=1,yrat=1):
def data_pull(file_loc):
    buffer, atts = ReadIM.extra.get_Buffer_andAttributeList(file_loc)
    v_array, _ = ReadIM.extra.buffer_as_array(buffer)
    data_mat = np.copy(v_array)
    shape = list(np.shape(data_mat))
    if len(shape) == 2:
        data_mat = data_mat.reshape(1,shape[0],shape[-1])
        shape = list(1,shape[0],shape[-1])
    else:
        pass
    n_figures = shape[0]
    attributes = np.copy(atts)
    del(v_array)
    del(buffer)
    del(atts)
    return data_mat,attributes,shape,n_figures

run_status = 0
if run_status == 1:

    n_contours = 50

    filename = file_loc.split("\\")[-1]
    cwd = file_loc.split(filename)[0]
    file_stripped = filename.split(".")[0]
    new_dir = file_stripped+"_Data"
    new_path = os.path.join(cwd,new_dir)
    try:
        os.mkdir(new_path,0o666)
    except:
        pass

    buffer, atts = ReadIM.extra.get_Buffer_andAttributeList(file_loc)

    new = ReadIM.extra.att2dict(atts)
    with open(new_path+"\\Attributes.txt", 'w') as f: 
        for key, value in new.items(): 
            while value[-1:] == "\n":
                value = value[:-1]
                f.write('%s:\n%s' % (key, value)+"\n\n")

    v_array, _ = ReadIM.extra.buffer_as_array(buffer)
    data_mat = np.copy(v_array)
    shape = list(np.shape(data_mat))
    data_mat = np.flip(data_mat,axis=1)

    length = len(shape) #assume x and y are always inputs
    if length == 2:
        n_fields = 1
    else:
        n_fields = shape[0]
    I = shape[-2]
    J = shape[-1]
    new_mat = np.ravel(data_mat).reshape(-1,n_fields,order='F')
    II,JJ = np.meshgrid(np.arange(1,I+1,1,dtype=int),np.arange(1,J+1,1,dtype=int))
    II_vec = II.reshape(I*J,1)
    JJ_vec = JJ.reshape(I*J,1)

    II_plot,JJ_plot = np.meshgrid(np.linspace(0,25,I),np.linspace(0,20,J))

    for index in range(n_fields):
        plt.figure(figsize=(16,8))
        ax = plt.gca()
        ax.set_aspect(0.5)
        u = np.copy(data_mat[index,:,:])
        if index==0:
            plt.title("u")
        if index==1:
            plt.title("v")
            u = -u
        plot_xmat = (II[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]-xmin_pixel)*xrat
        plot_ymat = (JJ[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]-ymin_pixel)*yrat
        plot_umat = u[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]
        plt.contourf(plot_xmat,plot_ymat,plot_umat,n_contours,cmap='jet')
        plt.colorbar()
        plot_save_index = new_path+"\\Plot_"+str(index)
        plt.savefig(new_path+"\\Plot_"+str(index),dpi=600)
        np.savetxt(new_path+"\\Data_field"+str(index)+".txt",u)
        data = u.reshape
        plt.close()
        del(u)
    del(data)
    del(v_array)



    if n_fields ==2:
        u_mag = (data_mat[0,:,:]**2+data_mat[1,:,:]**2)**(0.5)
        u_mag_vec = np.ravel(u_mag).reshape(-1,1,order='F')
        output_mat = np.concatenate(((II_vec-xmin_pixel)*xrat,(JJ_vec-ymin_pixel)*yrat,new_mat,u_mag_vec),axis=1)

        column_names = ["x (mm)","y (mm)"]
        for field in range(1,n_fields+1):
            column_name = "F(xy)_"+str(field)
            column_names.append(column_name)
        column_names.append("|V|")


        df = pd.DataFrame(output_mat,columns=column_names)
        df.to_csv(new_path+"\\Data.dat")
    else:
        output_mat = np.concatenate(((II_vec-xmin_pixel)*xrat,(JJ_vec-ymin_pixel)*yrat,new_mat),axis=1)

        column_names = ["x","y"]
        for field in range(1,n_fields+1):
            column_name = "F(xy)_"+str(field)
            column_names.append(column_name)

        df = pd.DataFrame(output_mat,columns=column_names)
        df.to_csv(new_path+"\\Data.dat")


    if n_fields ==2:

        shape = list(np.shape(data_mat))
        length = len(shape) #assume x and y are always inputs
        I = shape[-2]
        J = shape[-1]
        new_mat = np.ravel(data_mat).reshape(-1,n_fields,order='F')
        II,JJ = np.meshgrid(np.arange(1,I+1,1,dtype=int),np.arange(1,J+1,1,dtype=int))
        II_vec = II.reshape(I*J,1)
        JJ_vec = JJ.reshape(I*J,1)
        output_mat = np.concatenate((II_vec,JJ_vec,new_mat),axis=1)

        n_values = np.shape(output_mat)[0]
        xstep = int(I/50)
        #x_locations = np.arange(0,I-1,xstep,dtype=int)
        ystep = int(I/50)
        #y_locations = np.arange(0,J-1,ystep,dtype=int)
        x_locations = np.arange(xmin_pixel,xmax_pixel,xstep,dtype=int)
        y_locations = np.arange(ymin_pixel+2*ystep,ymax_pixel+ystep,ystep,dtype=int)
        
        
        
        indices = []
        for y_location in y_locations:
            new_vec = np.array(x_locations+y_location*I,dtype=int)
            indices = np.append(indices,new_vec,axis=0)
        indices = np.array(indices,dtype=int)
        scale_val = 125

        x_vals = np.ravel(np.rot90(II.T)).reshape(-1,1,order="F")[indices]-xmin_pixel
        x_vals = x_vals*xrat
        y_vals = np.ravel(np.rot90(JJ.T)).reshape(-1,1,order="F")[indices]-ymin_pixel
        y_vals = y_vals*yrat
        u_vals = np.ravel(np.rot90(data_mat[0,:,:].T)).reshape(-1,1,order="F")[indices]
        v_vals = np.ravel(np.rot90(data_mat[1,:,:].T)).reshape(-1,1,order="F")[indices]

        u_mag = (data_mat[0,:,:]**2+data_mat[1,:,:]**2)**(0.5)

        plt.figure(figsize=(16,8))
        ax = plt.gca()
        ax.set_aspect(0.5)
        plot_xmat = (II[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]-xmin_pixel)*xrat
        plot_ymat = (JJ[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]-ymin_pixel)*yrat
        plot_umat = u_mag[ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]
        plt.contourf(plot_xmat,plot_ymat,plot_umat,n_contours,cmap='jet')
        plt.colorbar()
        plt.quiver(x_vals,y_vals,u_vals,-v_vals,scale=scale_val)
        plt.title("|V|")

        
        plt.savefig(new_path+"\\Plot_V_mag",dpi=600)



        new_mat = np.ravel(data_mat[:,ymin_pixel:ymax_pixel,xmin_pixel:xmax_pixel]).reshape(-1,n_fields,order='F')

        output_mat = np.concatenate((np.ravel(plot_xmat).reshape(-1,1,order='F'),np.ravel(plot_ymat).reshape(-1,1,order='F'),new_mat,np.ravel(plot_umat).reshape(-1,1,order='F')),axis=1)

        column_names = ["x (mm)","y (mm)","u (m/s)","v (m/s)","|V| (m/s)"]
        # for field in range(1,n_fields+1):
        #     column_name = "F(xy)_"+str(field)
        #     column_names.append(column_name)
        # column_names.append("|V|")

        df = pd.DataFrame(output_mat,columns=column_names)
        df.to_csv(new_path+"\\Data_cropped.dat")
