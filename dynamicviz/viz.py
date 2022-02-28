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

def interactive(df, label, show=False, save=False, **kwargs):
    '''
    Generate Plotly interactive visualization
    
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str, column in output for labeling/coloring visualization
        show = True or False, whether to show the visualization
        save = False or str, path to save the visualization (acceptable extensions are .html)
        
    Returns:
        fig = Plotly object
    '''
    dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])

    fig = px.scatter(df, x='x1', y='x2', animation_frame="bootstrap_number", color=label, hover_name=label,
               range_x=[-dim,dim], range_y=[-dim,dim], **kwargs)
    
    if isinstance(save, str):
        if ".html" in save:
            fig.write_html(save)
        else:
            raise Exception("save needs to be .html file")
    
    if show is True:
        fig.show()
    
    return(fig)
    
    
def dynamic(df, label, save=False, **kwargs):
    '''
    Generate dynamic visualization
    
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str, column in output for labeling/coloring visualization
        save = False or str, path to save the visualization (acceptable extensions are .gif and .avi)
        
    Returns:
        images = list of matplotlib objects
    '''
    dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])
    
    # generate list of matplotlib plots
    images = []
    for boot_num in np.unique(df["bootstrap_number"]):
        sub_df = df[df["bootstrap_number"] == boot_num].reset_index()
        fig = plt.figure(figsize=(4,4))
        if df[label].dtype == np.float:
            plt.scatter(sub_df["x1"],sub_df["x2"],c=sub_df[label], **kwargs)
            plt.colorbar()
        else:
            for lab in np.unique(df[label]):
                plt.scatter(sub_df[sub_df[label] == lab]["x1"], sub_df[sub_df[label] == lab]["x2"],
                            label=str(lab), **kwargs)
        plt.ylim(-dim,dim)
        plt.xlim(-dim,dim)
        plt.tight_layout()
        im = fig2img(fig)
        images.append(im)
        plt.close()
    
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
    
    return(images)
    
    
def static(df, label, show=False, save=False, **kwargs):
    '''
    Generate static visualization
    
    Arguments:
        df = pandas dataframe corresponding to output of boot.generate() 
        label = str, column in output for labeling/coloring visualization
        show = True or False, whether to show the visualization
        save = False or str, path to save the visualization (acceptable extensions are .png, .jpg, .pdf, .eps, .tiff, ...
        
    Returns:
        fig = matplotlib object
    '''
    dim = np.max([np.max(np.abs(df['x1'])),np.max(np.abs(df['x2']))])
    alpha = 0.2/(df.shape[0]/1000)

    fig = plt.figure(figsize=(4,4))
    if df[label].dtype == np.float:
        plt.scatter(df["x1"],df["x2"], c=df[label], alpha=alpha, edgecolors='none', **kwargs)
        color_bar = plt.colorbar()
        color_bar.solids.set(alpha=1)
    else:
        for lab in np.unique(df[label]):
            plt.scatter(df[df[label] == lab]["x1"],df[df[label] == lab]["x2"],
                        label=str(lab), alpha=alpha, edgecolors='none', **kwargs)
            leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
    plt.ylim(-dim,dim)
    plt.xlim(-dim,dim)
    plt.tight_layout()
    if isinstance(save, str):
        plt.savefig(save, dpi=500, bbox_inches = "tight")
    if show is True:
        plt.show()
    
    return(fig)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=500)
    buf.seek(0)
    img = Image.open(buf)
    return img
    
    
    
