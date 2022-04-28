"""
Dynamic visualization viewing modality module
- Tools for creating different viewing modalities for the dynamic visualizations
"""

# author: Eric David Sun <edsun@stanford.edu>
# (C) 2022 
from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px



def interactive(df, label, show=False, save=False, colors=None,
            xlabel="x1", ylabel="x2", title="", legend_title="",
            vmin=None, vmax=None, alpha=1.0, width=4, height=4, dpi=500, dim=None):
    '''
    Generate dynamic visualization with interactive (plotly) modality
    
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str or float, column in output for labeling/coloring visualization
        show = True or False, whether to show the visualization
        save = False or str, path to save the visualization (acceptable extensions are .html)
        colors = list of plotly color codes to use if labels are discrete
        xlabel = label of x-axis
        ylabel = label of y-axis
        title = title of plot
        legend_title = title of legend
        vmin = set value of vmin for continuous colorbar (default is min across all bootstrap frames)
        vmax = set value of vmax for continuous colorbar (default is max across all bootstrap frames)
        alpha = opacity between 0 and 1
        width = width of plotly plot
        height = height of plotly plot
        dpi = dots per inch
        dim = x and y dimensions will +- dim in either direction; default is to take the maximum across all frames
        
        will convert width, height, dpi into plotly.express.scatter() width and height
        
    Returns:
        fig = Plotly object
    '''
    # run some checks
    if isinstance(colors, list):
        if len(colors) < len(np.unique(df[label])):
            raise Exception ("colors need to have >= length as len(np.unique(df[label]))")
    
    # get bootstrap-wide statistics for consistency across frames
    if dim is None:
        dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])
    
    
    if df[label].dtype == np.float: # define max and min values across all frames
        if vmin is None:
            vmin = np.min(df[label])
        if vmax is None:
            vmax = np.max(df[label])
            
    # compute width and height
    width = width * dpi
    height = height * dpi
    
                
    df_px = df.copy() # make copy of df
    
    # create color dict if needed
    if (df_px[label].dtype != np.float) and (isinstance(colors, list)):
        color_dict = {}
        for li, lab in enumerate(np.unique(df_px[label])):
            color_dict[lab] = colors[li]
        
    # add new columns for marker
    #df_px['symbol'] = marker
    
    # make figure
    if df[label].dtype == np.float: # continuous labels
        fig = px.scatter(df_px, x='x1', y='x2', animation_frame="bootstrap_number", animation_group="original_index", color=label, hover_name=label, title=title,
                   range_x=[-dim,dim], range_y=[-dim,dim], range_color=(vmin,vmax), opacity=alpha, width=width, height=height,
                   labels={
                         "x1": xlabel,
                         "x2": ylabel,
                         label: legend_title
                     })
    elif isinstance(colors, list): # if colors is specified for discrete labels           
        fig = px.scatter(df_px, x='x1', y='x2', animation_frame="bootstrap_number", animation_group="original_index", color=label, hover_name=label, title=title,
                   range_x=[-dim,dim], range_y=[-dim,dim], opacity=alpha, width=width, height=height,
                   color_discrete_map = color_dict,
                   labels={
                         "x1": xlabel,
                         "x2": ylabel,
                         label: legend_title
                     })
                     
    else: # discrete labels (default colors)
        fig = px.scatter(df_px, x='x1', y='x2', animation_frame="bootstrap_number", animation_group="original_index", color=label, hover_name=label, title=title,
                   range_x=[-dim,dim], range_y=[-dim,dim], opacity=alpha, width=width, height=height,
                   labels={
                         "x1": xlabel,
                         "x2": ylabel,
                         label: legend_title
                     })

    if isinstance(save, str): # save as HTML
        if ".html" in save:
            fig.write_html(save)
        else:
            raise Exception("save needs to be .html file")
    
    if show is True: # show figure
        fig.show()
    
    return(fig)
    
    
def animated(df, label, save=False, get_frames=None, colors=None, cmap='viridis', width=4, height=4, dpi=500,
            xlabel=None, ylabel=None, title=None, title_fontsize=16, major_fontsize=16, minor_fontsize=14,
            vmin=None, vmax=None, marker="o", alpha=1.0, solid_cbar=True, show_legend=True, solid_legend=True,
            legend_fontsize=12, dim=None, **kwargs):
    '''
    Generate dynamic visualization with animated modality
        
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str, column in output for labeling/coloring visualization
        save = False or str, path to save the visualization (acceptable extensions are .gif and .avi)
        get_frames = None or list of integer indices to save frames for in folder called save.split(".")[0]
        colors = list of matplotlib color codes to use if labels are discrete
        cmap = color map to use for continuous labels
        width = width of matplotlib plot
        height = height of matplotlib plot
        dpi = dots per inch
        xlabel = label of x-axis
        ylabel = label of y-axis
        title = title of plot
        title_fontsize = fontsize for title
        major_fontsize = fontsize for axis labels
        minor_fontsize = fontsize for axis ticks
        vmin = set value of vmin for continuous colorbar (default is min across all bootstrap frames)
        vmax = set value of vmax for continuous colorbar (default is max across all bootstrap frames)
        marker = marker symbol for plot
        alpha = opacity between 0 and 1
        solid_cbar = boolean, whether to set alpha=1 for color bar
        show_legend = boolean, whether to show a legend
        solid_legend = boolean, whether to set alpha=1 for legend markers
        legend_fontsize = fontsize for legend
        dim = x and y dimensions will +- dim in either direction; default is to take the maximum across all frames
                
    Returns:
        images = list of matplotlib objects
    '''
    # run some checks
    if isinstance(colors, list):
        if len(colors) < len(np.unique(df[label])):
            raise Exception ("colors need to have >= length as len(np.unique(df[label]))")
    
    # get bootstrap-wide statistics for consistency across frames
    if dim is None:
        dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])
    if df[label].dtype == np.float:
        if vmin is None:
            vmin = np.min(df[label])
        if vmax is None:
            vmax = np.max(df[label])
    
    # generate list of matplotlib plots
    images = []
    for boot_num in np.sort(np.unique(df["bootstrap_number"])):
    
        # get bootstrap results
        sub_df = df[df["bootstrap_number"] == boot_num].reset_index()
        
        # make figure
        fig = plt.figure(figsize=(width, height))
        
        # continuous labels for float labels
        if df[label].dtype == np.float:
            plt.scatter(sub_df["x1"], sub_df["x2"], c=sub_df[label], cmap=cmap, vmin=vmin, vmax=vmax,
                        marker=marker, alpha=alpha, **kwargs)
            cbar = plt.colorbar()
            if solid_cbar is True:
                cbar.solids.set(alpha=1)
            
        # otherwise, discrete labels
        else:
            for li, lab in enumerate(np.unique(df[label])):
                if colors is None:
                    plt.scatter(sub_df[sub_df[label] == lab]["x1"], sub_df[sub_df[label] == lab]["x2"],
                                label=str(lab), marker=marker, alpha=alpha, **kwargs)
                else:
                    plt.scatter(sub_df[sub_df[label] == lab]["x1"], sub_df[sub_df[label] == lab]["x2"],
                                label=str(lab), c=colors[li], marker=marker, alpha=alpha, **kwargs)
                                
            if show_legend is True:
                leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
                if solid_legend is True:
                    for lh in leg.legendHandles: 
                        lh.set_alpha(1)
        
        # add titles and axis labels
        if isinstance(title, str):
            plt.title(title, fontsize=title_fontsize)
        if isinstance(xlabel, str):
            plt.xlabel(xlabel, fontsize=major_fontsize)
        if isinstance(ylabel, str):
            plt.ylabel(ylabel, fontsize=major_fontsize)
            
        plt.xticks(fontsize=minor_fontsize)
        plt.yticks(fontsize=minor_fontsize)
            
        # specify plot parameters
        plt.ylim(-dim,dim)
        plt.xlim(-dim,dim)
        plt.tight_layout()
        im = fig2img(fig, dpi=dpi)
        images.append(im)
        plt.close()
    
    # save dynamic visualization
    if isinstance(save, str):
        # save as GIF
        if ".gif" in save:
            images[0].save(save, save_all = True, append_images = images[1:], 
                           optimize = False, duration = [1000]+[100]*len(images[1:]), loop=0)
        # save as AVI
        elif ".avi" in save:
            frame = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            height, width, layers = frame.shape
            video = cv2.VideoWriter(save, 0, 10, (width,height))
            for image in images:
                opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                video.write(opencvImage)
            cv2.destroyAllWindows()
            video.release()
        else:
            raise Exception("save needs to be .gif or .avi file")
    
    # save still frames if specified
    if isinstance(get_frames, list):
        if not os.path.exists(save.split(".")[0]):
            os.makedirs(save.split(".")[0])
        for index in get_frames:
            im = images[index]
            im.save(os.path.join(save.split(".")[0],"frame"+str(index)+".png"))
    
    return(images)
    
    
def stacked(df, label, show=False, save=False, colors=None, cmap='viridis', width=4, height=4, dpi=500,
            xlabel=None, ylabel=None, title=None, title_fontsize=16, major_fontsize=16, minor_fontsize=14,
            vmin=None, vmax=None, marker="o", alpha=None, solid_cbar=True, show_legend=True, solid_legend=True,
            legend_fontsize=12, dim=None, frame=None, return_fig=True, **kwargs):
    '''
    Generate dynamic visualization with stacked modality
        
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str, column in output for labeling/coloring visualization
        show = True or False, whether to show the visualization
        save = False or str, path to save the visualization (acceptable extensions are .png, .jpg, .pdf, .eps, .tiff, ...
        frame = None or if int, then shows only that bootstrap frame
        return_fig = boolean, whether to return figure or not
        
        See animated() for more details
                
    Returns:
        fig = matplotlib object
    '''
    # run some checks
    if isinstance(colors, list):
        if len(colors) < len(np.unique(df[label])):
            raise Exception ("colors need to have >= length as len(np.unique(df[label]))")
    
    # get bootstrap-wide statistics for consistency across frames
    if dim is None:
        dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])
    if df[label].dtype == np.float:
        if vmin is None:
            vmin = np.min(df[label])
        if vmax is None:
            vmax = np.max(df[label])
        
    # automatically set alpha based on heuristic
    if alpha is None:
        alpha = 0.2/(df.shape[0]/1000)

    fig = plt.figure(figsize=(width,height))
    
    # snapshot frame if requested
    if isinstance(frame, int):
        df = df[df['bootstrap_number']==frame]
    
    # continuous labels
    if df[label].dtype == np.float:
        plt.scatter(df["x1"],df["x2"], c=df[label], cmap=cmap, vmin=vmin, vmax=vmax,
                        marker=marker, alpha=alpha, edgecolors='none', **kwargs)
        cbar = plt.colorbar()
        if solid_cbar is True:
            cbar.solids.set(alpha=1)
    
    # discrete labels
    else:
        for li, lab in enumerate(np.unique(df[label])):
            if colors is None:
                plt.scatter(df[df[label] == lab]["x1"],df[df[label] == lab]["x2"],
                            label=str(lab), marker=marker, alpha=alpha, edgecolors='none', **kwargs)
            else:
                plt.scatter(df[df[label] == lab]["x1"],df[df[label] == lab]["x2"],
                            label=str(lab), c=colors[li], marker=marker, alpha=alpha, edgecolors='none', **kwargs)

        if show_legend is True:
            leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
            if solid_legend is True:
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)
                    
    # add titles and axis labels
    if isinstance(title, str):
        plt.title(title, fontsize=title_fontsize)
    if isinstance(xlabel, str):
        plt.xlabel(xlabel, fontsize=major_fontsize)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel, fontsize=major_fontsize)
        
    plt.xticks(fontsize=minor_fontsize)
    plt.yticks(fontsize=minor_fontsize)
                    
    plt.ylim(-dim,dim)
    plt.xlim(-dim,dim)
    plt.tight_layout()
    
    # save plot
    if isinstance(save, str):
        plt.savefig(save, dpi=dpi, bbox_inches = "tight")
    if show is True:
        plt.show()
    else:
        plt.close()
    
    if return_fig is True:
        return(fig)
    


def fig2img(fig, dpi):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img
    
    
    
