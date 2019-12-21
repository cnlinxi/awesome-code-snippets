# Configure My Jupyter Notebook

### Modify Workspace Dir

- In Shell:

    `jupyter notebook --generate-config`

- Edit `jupyter_notebook_config.py`:

    ```Python
    ## The directory to use for notebooks and kernels. 
    c.NotebookApp.notebook_dir = '<Your Notebook Dir>'
    ```

    to your dir.

### Extension

- Install Extension Manager in Shell:
  
  `pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install`

- Start Jupyter Notebook and navigate to the `Nbextensions` tab:

    ![](https://image.jiqizhixin.com/uploads/editor/0c4fdc3d-f8b9-4c0e-87f1-2be29a44e1ef/1545285489870.png)
  
  Select:

  - `Hinterland`: Enable code autocompletion menu for every keypress in a code cell, instead of only enabling it with tab
  
  - `Autopep8`: Use kernel-specific code to reformat/prettify the contents of code cells
  
    - Prerequisites: `pip install autopep8`
  
    - Modify Hotkey to use to prettify the selected cell(s) to `Alt-Ctrl-L`

### Modify Notebook Style

- Navigate to Python Package Dir: `.../Lib/site-packages/notebook/static/custom/`

- Edit `custom.css`. An example:

    ```css
    /*
    Placeholder for custom user CSS

    mainly to be overridden in profile/static/custom/custom.css

    This will always be an empty file in IPython
    */

    /*for the error , connecting & renaming window*/

    .CodeMirror pre {font-family: Monaco; font-size: 10pt;}
    * {font-family: Monaco;}
    div.output_area pre {font-family: Monaco; font-size: 10pt;}
    div.input_prompt {font-family: Monaco; font-size: 10pt;}
    div.out_prompt_overlay {font-family: Monaco; font-size: 10pt;}
    div.prompt {font-family: Monaco; font-size: 10pt;}
    span.cm-comment {font-family:  Monaco !important; font-style:normal !important; color:#FFAE3C !important;}
    ```
