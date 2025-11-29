"""
Visualization module for pyehicle.

This module provides functions to visualize GPS trajectories on interactive maps
using Folium or static plots using matplotlib.
"""

import numpy as np
import pandas as pd
import polars as pl
import folium
from matplotlib import pyplot as plt, cm
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
from branca.element import Template, MacroElement


def single(df: pd.DataFrame | pl.DataFrame,
           name: str = None,
           return_map: bool = False,
           show_legend: bool = True,
           legend_name: str = 'Color by Trajectory',
           show_in_browser: bool = True,
           lat_col: str = 'lat',
           lon_col: str = 'lon',
           tiles: str = "OpenStreetMap"):
    """
    Plot a single trajectory on an interactive Folium map.

    This function creates an interactive map visualization of a GPS trajectory with
    customizable appearance and legend. The map is centered on the trajectory's centroid.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The trajectory data to plot with latitude and longitude columns.
    name : str, optional
        The name of the trajectory to display in the legend. Defaults to "Trajectory".
    return_map : bool, default=False
        If True, returns the folium.Map object instead of displaying it.
    show_legend : bool, default=True
        If True, displays a draggable legend on the map.
    legend_name : str, default='Color by Trajectory'
        The title of the legend box.
    show_in_browser : bool, default=True
        If True, opens the map in the default web browser.
    lat_col : str, default='lat'
        The column name for latitude values.
    lon_col : str, default='lon'
        The column name for longitude values.
    tiles : str, default="OpenStreetMap"
        The map tile style. Options include: "OpenStreetMap", "Stamen Terrain",
        "Stamen Toner", "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter".

    Returns
    -------
    folium.Map or None
        If return_map=True, returns the folium.Map object. Otherwise, returns None.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>> df = pd.read_csv('trajectory.csv')
    >>> pye.utilities.visualization.single(df, name='My Route', show_in_browser=True)

    >>> # Use different map tiles
    >>> pye.utilities.visualization.single(df, tiles="CartoDB positron")

    >>> # Get map object for further customization
    >>> map_obj = pye.utilities.visualization.single(df, return_map=True, show_in_browser=False)
    >>> # Add custom markers or layers to map_obj
    >>> map_obj.show_in_browser()

    Notes
    -----
    The trajectory is drawn as an orange polyline with weight=3.
    The map is automatically centered on the mean coordinates of the trajectory
    with a default zoom level of 15 (suitable for city-level detail).
    """
    # Extract coordinates as numpy arrays (handles both pandas and polars)
    if isinstance(df, pl.DataFrame):
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()
    else:
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()

    # Calculate map center from trajectory centroid
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # Create base map centered on trajectory
    _map = folium.Map([center_lat, center_lon], zoom_start=15, tiles=tiles)

    # Create trajectory polyline (orange color with thickness 3)
    points = list(zip(lats.tolist(), lons.tolist()))
    folium.PolyLine(points, color="orange", weight=3).add_to(_map)

    # Prepare legend items (name, color pairs)
    items = [(name if name is not None else "Trajectory", "orange")]

    # Add legend and layer control if requested
    if show_legend:
        __add_map_legend(_map, legend_name, items)
        folium.map.LayerControl().add_to(_map)

    # Return map object if requested
    if return_map:
        return _map

    # Display in browser if requested
    if show_in_browser:
        _map.show_in_browser()

    return None


def multiple(
    df_list: list,
    names: list = None,
    return_map: bool = False,
    show_legend: bool = True,
    legend_name: str = 'Color by Trajectory',
    show_in_browser: bool = True,
    cmap: str = 'tab10',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    tiles: str = "OpenStreetMap"
):
    """
    Plot multiple trajectories on a single interactive Folium map.

    This function visualizes multiple GPS trajectories on the same map with different
    colors for easy comparison. Each trajectory is drawn as a polyline with automatic
    color assignment from the specified colormap.

    Parameters
    ----------
    df_list : list of pd.DataFrame or pl.DataFrame
        List of trajectory DataFrames to plot. Each DataFrame must contain
        latitude and longitude columns.
    names : list of str, optional
        List of names for the trajectories to display in the legend.
        If None, trajectories are labeled as "Trajectory 1", "Trajectory 2", etc.
        If the list is shorter than df_list, remaining trajectories get default names.
    return_map : bool, default=False
        If True, returns the folium.Map object instead of displaying it.
    show_legend : bool, default=True
        If True, displays a draggable legend on the map showing trajectory names and colors.
    legend_name : str, default='Color by Trajectory'
        The title of the legend box.
    show_in_browser : bool, default=True
        If True, opens the map in the default web browser.
    cmap : str, default='tab10'
        Matplotlib colormap name for trajectory colors. Popular options include:
        'tab10', 'Set1', 'Set2', 'Paired', 'Accent', 'viridis', 'plasma'.
    lat_col : str, default='lat'
        The column name for latitude values in all DataFrames.
    lon_col : str, default='lon'
        The column name for longitude values in all DataFrames.
    tiles : str, default="OpenStreetMap"
        The map tile style. Options include: "OpenStreetMap", "Stamen Terrain",
        "Stamen Toner", "Stamen Watercolor", "CartoDB positron", "CartoDB dark_matter".

    Returns
    -------
    folium.Map or None
        If return_map=True, returns the folium.Map object. Otherwise returns None.

    Raises
    ------
    ValueError
        If df_list is empty, if any DataFrame is missing required columns,
        or if no coordinates are available for centering the map.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> raw = pd.read_csv('raw_trajectory.csv')
    >>> matched = pd.read_csv('matched_trajectory.csv')
    >>> reconstructed = pd.read_csv('reconstructed_trajectory.csv')
    >>>
    >>> # Compare three trajectories with custom names
    >>> pye.utilities.visualization.multiple(
    ...     df_list=[raw, matched, reconstructed],
    ...     names=['Raw GPS', 'Map-Matched', 'Reconstructed'],
    ...     show_in_browser=True
    ... )

    >>> # Use different colormap and map style
    >>> pye.utilities.visualization.multiple(
    ...     df_list=[traj1, traj2, traj3],
    ...     cmap='viridis',
    ...     tiles='CartoDB dark_matter'
    ... )

    >>> # Get map object for further customization
    >>> map_obj = pye.utilities.visualization.multiple(
    ...     df_list=[traj1, traj2],
    ...     names=['Before', 'After'],
    ...     return_map=True,
    ...     show_in_browser=False
    ... )

    Notes
    -----
    - Trajectories are drawn with weight=3 and opacity=0.7 for better visibility
      when multiple trajectories overlap.
    - The map is automatically centered on the centroid of all trajectories combined.
    - Colors are automatically assigned from the specified colormap to ensure
      visual distinction between trajectories.
    """
    # Validate input
    if not df_list:
        raise ValueError("df_list is empty. Provide at least one DataFrame to plot.")

    # Collect all coordinates from all DataFrames to calculate map center
    all_lats_list = []
    all_lons_list = []

    for df in df_list:
        # Check if required columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"DataFrame missing required columns: '{lat_col}' and/or '{lon_col}'.")

        # Extract coordinates as numpy arrays (handles both pandas and polars)
        if isinstance(df, pl.DataFrame):
            all_lats_list.append(df[lat_col].to_numpy())
            all_lons_list.append(df[lon_col].to_numpy())
        else:
            all_lats_list.append(df[lat_col].to_numpy())
            all_lons_list.append(df[lon_col].to_numpy())

    # Combine all coordinate arrays to find global centroid
    all_lats = np.concatenate(all_lats_list) if all_lats_list else np.array([])
    all_lons = np.concatenate(all_lons_list) if all_lons_list else np.array([])

    # Validate that we have coordinates to work with
    if len(all_lats) == 0 or len(all_lons) == 0:
        raise ValueError("No coordinates available to center the map.")

    # Calculate map center from combined coordinates
    initial_lat = float(np.nanmean(all_lats))
    initial_lon = float(np.nanmean(all_lons))

    # Create base map centered on all trajectories
    _map = folium.Map([initial_lat, initial_lon], zoom_start=15, tiles=tiles)

    # Plot trajectories with different colors
    items = []  # Store (name, color) pairs for legend

    # Get colormap for trajectory color assignment
    _cmap = cm.get_cmap(cmap)
    num_dfs = len(df_list)

    # Iterate through each trajectory DataFrame
    count = 1
    for df in df_list:
        # Extract coordinate arrays (handles both pandas and polars)
        if isinstance(df, pl.DataFrame):
            lats_arr = df[lat_col].to_numpy()
            lons_arr = df[lon_col].to_numpy()
        else:
            lats_arr = df[lat_col].to_numpy()
            lons_arr = df[lon_col].to_numpy()

        # Create list of (lat, lon) tuples for Folium PolyLine
        points_latlon = list(zip(lats_arr, lons_arr))

        # Assign color from colormap (normalize by number of trajectories)
        color_ = rgb2hex(_cmap(count / max(1, num_dfs)))

        # Add polyline to map with opacity for overlapping trajectories
        folium.PolyLine(points_latlon, color=color_, weight=3, opacity=0.7).add_to(_map)

        # Store trajectory name and color for legend
        trajectory_name = names[count-1] if names and count-1 < len(names) else f"Trajectory {count}"
        items.append((trajectory_name, color_))
        count += 1

    # Add legend if requested
    if show_legend:
        # Calculate legend height based on number of items
        legend_height = 25 * (len(items) + 1) + 20

        # Build legend HTML
        legend_html = f'''
             <div style="position: fixed;
             bottom: 50px; left: 50px; width: 220px; height: {legend_height}px;
             border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;">
             &nbsp;<b>{legend_name}</b><br>
        '''
        # Add each trajectory to the legend
        for name, color in items:
            legend_html += f'&nbsp;<i class="fa fa-circle" style="color:{color}"></i>&nbsp;{name}<br>'
        legend_html += '</div>'

        # Add legend to map
        _map.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control for toggling layers
    folium.LayerControl().add_to(_map)

    # Return map object if requested
    if return_map:
        return _map

    # Display in browser if requested
    if show_in_browser:
        _map.show_in_browser()

    return None


def single_plt(df: pd.DataFrame | pl.DataFrame,
               lat_col: str = 'lat',
               lon_col: str = 'lon',
               name: str = 'Trajectory'):
    """
    Plot a single trajectory on a static matplotlib plot.

    This function creates a simple 2D line plot of a GPS trajectory using matplotlib.
    Unlike the Folium-based `single()` function, this creates a static image suitable
    for publications, reports, or quick visualization without a web browser.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The trajectory data to plot with latitude and longitude columns.
    lat_col : str, default='lat'
        The column name for latitude values.
    lon_col : str, default='lon'
        The column name for longitude values.
    name : str, default='Trajectory'
        The name of the trajectory for the legend.

    Returns
    -------
    None
        Displays the plot using plt.show().

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>> df = pd.read_csv('trajectory.csv')
    >>> pye.utilities.visualization.single_plt(df, name='My Route')

    Notes
    -----
    - The trajectory is plotted in orange color with a solid line.
    - This is a static plot (not interactive) suitable for saving to files.
    - The plot uses raw lon/lat coordinates (not projected), which may appear
      distorted for large areas or high latitudes.
    - To save the plot, use plt.savefig() before calling this function's plt.show().
    """
    # Extract coordinates as numpy arrays (handles both pandas and polars)
    if isinstance(df, pl.DataFrame):
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()
    else:
        lats = df[lat_col].to_numpy()
        lons = df[lon_col].to_numpy()

    # Plot the trajectory polyline (lon on x-axis, lat on y-axis)
    plt.plot(lons, lats, color='orange', linestyle='solid')

    # Add axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Add legend
    plt.legend([name])

    # Display the plot
    plt.show()


def multiple_plt(df_list: list,
                 lat_col: str = 'lat',
                 lon_col: str = 'lon',
                 names: list = None):
    """
    Plot multiple trajectories on a single static matplotlib plot.

    This function creates a simple 2D line plot of multiple GPS trajectories using
    matplotlib. Unlike the Folium-based `multiple()` function, this creates a static
    image suitable for publications, reports, or quick visualization without a web browser.

    Parameters
    ----------
    df_list : list of pd.DataFrame or pl.DataFrame
        List of trajectory DataFrames to plot. Each DataFrame must contain
        latitude and longitude columns.
    lat_col : str, default='lat'
        The column name for latitude values in all DataFrames.
    lon_col : str, default='lon'
        The column name for longitude values in all DataFrames.
    names : list of str, optional
        List of names for the trajectories to display in the legend.
        If None, trajectories are labeled as "Trajectory 1", "Trajectory 2", etc.
        If the list is shorter than df_list, missing names are auto-generated.

    Returns
    -------
    None
        Displays the plot using plt.show().

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> raw = pd.read_csv('raw_trajectory.csv')
    >>> matched = pd.read_csv('matched_trajectory.csv')
    >>> reconstructed = pd.read_csv('reconstructed_trajectory.csv')
    >>>
    >>> # Compare three trajectories
    >>> pye.utilities.visualization.multiple_plt(
    ...     df_list=[raw, matched, reconstructed],
    ...     names=['Raw GPS', 'Map-Matched', 'Reconstructed']
    ... )

    >>> # Quick comparison without custom names
    >>> pye.utilities.visualization.multiple_plt([traj1, traj2, traj3])

    Notes
    -----
    - Trajectories are automatically colored using the 'tab10' colormap.
    - This is a static plot (not interactive) suitable for saving to files.
    - The plot uses raw lon/lat coordinates (not projected), which may appear
      distorted for large areas or high latitudes.
    - To save the plot: plt.savefig('output.png') before calling this function.
    """
    # Create a new figure and axis
    _, ax = plt.subplots()

    # Prepare names list for legend
    num_dfs = len(df_list)
    if names is None:
        # Auto-generate names if not provided
        names = [f'Trajectory {i + 1}' for i in range(num_dfs)]
    elif len(names) < num_dfs:
        # Extend names list with auto-generated names if too short
        names = names + [f'Trajectory {i + 1}' for i in range(len(names), num_dfs)]

    # Get colormap for trajectory color assignment
    cmap = get_cmap('tab10')

    # Iterate over the list of DataFrames and plot each trajectory
    for count, df in enumerate(df_list, start=1):
        # Extract coordinates as numpy arrays (handles both pandas and polars)
        if isinstance(df, pl.DataFrame):
            lats = df[lat_col].to_numpy()
            lons = df[lon_col].to_numpy()
        else:
            lats = df[lat_col].to_numpy()
            lons = df[lon_col].to_numpy()

        # Plot the trajectory polyline with color from colormap
        ax.plot(lons, lats, color=cmap(count / num_dfs), linestyle='solid')

    # Add legend with trajectory names
    ax.legend(names)

    # Add axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Display the plot
    plt.show()


def __add_map_legend(m, title, items):
    """
    Add a draggable custom legend to a Folium map (internal helper function).

    This helper function creates a styled, draggable legend box for Folium maps.
    The legend displays trajectory names with colored circles as markers.

    Parameters
    ----------
    m : folium.Map
        The Folium map object to add the legend to.
    title : str
        The title text for the legend box.
    items : list of tuple
        List of (name, color) tuples where:
        - name (str): The label to display
        - color (str): The hex color code for the marker (e.g., '#FF5733')

    Notes
    -----
    The legend is:
    - Positioned in the bottom-right corner by default
    - Draggable via jQuery UI
    - Styled with rounded borders and semi-transparent background
    - Uses Font Awesome circles as color markers

    This is an internal function and should not be called directly by users.
    It is automatically invoked by single() and multiple() functions when
    show_legend=True.
    """
    # Build legend HTML items using template
    item_template = "<li><span style='background:{};'></span>{}</li>"

    # Create list of legend items from (name, color) tuples
    list_items = '\n'.join([item_template.format(c, n) for (n, c) in items])

    # HTML template with jQuery UI for draggable legend
    template = """
    {{% macro html(this, kwargs) %}}
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet"
        href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
      <script>
      $( function() {{
        $( "#maplegend" ).draggable({{
                        start: function (event, ui) {{
                            $(this).css({{
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            }});
                        }}
                    }});
    }});
      </script>
    </head>
    <body>
    <div id='maplegend' class='maplegend'
        style='position: absolute; z-index:9999; border:2px solid grey;
        background-color:rgba(255, 255, 255, 0.8); border-radius:6px;
        padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
    <div class='legend-title'> {} </div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        {}
      </ul>
    </div>
    </div>
    </body>
    </html>
    <style type='text/css'>
      .maplegend .legend-title {{
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }}
      .maplegend .legend-scale ul {{
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }}
      .maplegend .legend-scale ul li {{
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }}
      .maplegend ul.legend-labels li span {{
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }}
      .maplegend .legend-source {{
        font-size: 80%;
        color: #777;
        clear: both;
        }}
      .maplegend a {{
        color: #777;
        }}
    </style>
    {{% endmacro %}}""".format(
        title, list_items
    )

    # Create macro element and add to map
    macro = MacroElement()
    macro._template = Template(template)
    m.get_root().add_child(macro, name='map_legend')
