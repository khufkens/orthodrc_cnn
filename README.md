<style>
.legend {
	text-align: left;
	line-height: 18px;
	color: #555;
	padding: 6px 8px;
	font: 16px/18px Arial, Helvetica, sans-serif;
	background: rgba(255,255,255,0.8);
	box-shadow: 0 0 15px rgba(0,0,0,0.2);
	border-radius: 5px;
}

.legend h4 {
    margin: 0 0 5px;
	color: #777;
}

.legend i {
	width: 18px;
	height: 18px;
	float: left;
	margin-right: 8px;
	opacity: 0.7;
}

.legend .circle {
	border-radius: 50%;
	width: 10px;
	height: 10px;
	margin-top: 8px;
}


img {
  border-radius: 0%;
}

</style>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.3.4/dist/leaflet.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.10.4/jquery-ui.min.js"></script>
<script src='https://api.mapbox.com/mapbox.js/plugins/leaflet-fullscreen/v1.0.1/Leaflet.fullscreen.min.js'></script>
<link href='https://api.mapbox.com/mapbox.js/plugins/leaflet-fullscreen/v1.0.1/leaflet.fullscreen.css' rel='stylesheet' />


# Ortho DRC Deep Learning forest classifier

This is the code used to train the model (src/forest_model.h5) to classify forest and non forest pixels and a 1958 historical orthomosaic covering a large section of Yangambi in the Central Congo Basin, DR Congo (then Belgian Congo).

When using the model cite the model / code / data as:

`Hufkens et al. 2020. Historical aerial surveys map long-term changes of
forest cover and structure in the central Congo Basin. Submitted.`

## Methods

 We used the Unet Convolutional Neural Net (CNN) architecture implemented in Keras with an [efficientnetb3 backbone](https://github.com/qubvel/segmentation_models) running on TensorFlow to train a binary classifier (i.e. forest or non-forested). This methodology is increasingly being used to automate pixel-level classification in (color) digital photography data. Training data were collected from the orthomosaic by randomly selecting 513 pixel square tiles from homogeneous forested or non-forested areas within the historical orthomosaic.
 
 Homogeneous tiles were combined in synthetic landscapes using a random gaussian field based binary mask (Figure 3). We generated 5000 synthetic landscapes for training, while 500 landscapes were generated for both the validation and the testing dataset. Source tiles did not repeat across datasets to limit overfitting. In order to limit stitch line misclassifications, along the seams of mosaicked images, I created synthetic landscapes with different forest tiles to mimick forest texture transitions. I applied this technique to 10% of the generated synthetic landscapes. In order to limit the size of the repository the generated landscapes are not included. However, and example of a synthetic landscape and a forest / non-forest mask is provided below. A new dataset can be generated using the included code.
 
![](synthetic_landscape.png) 
 
 The CNN model was trained for 100 epochs on a graphics processing unit (GPU) maximizing the Intersect-over-Union (IoU) using additional data augmentation. Data augmentation included random cropping to 320 pixel squares, random orientation, scaling, perspective, contrast and brightness shifts and image blurring. During final model evaluation we report the IoU of our out-of-sample test datasets.  The optimized model was used to classify the complete orthomosaic using a moving window approach with a step size of 110 pixels and a majority vote across overlapping areas to limit segmentation edge effects. I refer to the figure below for a synoptic overview of the full deep learning learning workflow. 

![](cnn_diagram.png) 

## Results

A scrollable orthomosaic composite of aerial photos and the resulting forest cover map is provided below.

<div id="map" style="width: 600px%; height: 600px; z-index:0;"></div>

## Licenses & citation

I'm indebted to the [segmentation models](https://github.com/qubvel/segmentation_models) python package and examples from which I extensively borrowed code. To be inline with licensing all code is distributed under an MIT license. All other (image) data, and the resulting model (forest_model.h5) is distributed under a CC-BY-NC-SA license.

## Acknowledgements

This research was supported through the Belgian Science Policy office COBECORE project (BELSPO; grant BR/175/A3/COBECORE) and from the European Union Marie Skłodowska-Curie Action (project number 797668).


<script>
      var map = L.map('map').setView([0.9, 24.5], 13);
      var baselayer =  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{
    	maxZoom: 16,
    	minZoom: 13,
    	subdomains:['mt0']}).addTo(map);
	var ortho = L.tileLayer('https://github.com/khufkens/COBECORE_maps/raw/master/ortho/{z}/{x}/{y}.png', {
        maxZoom: 16,
	    minZoom: 13,
        tms: false
      }).addTo(map);
      var cover = L.tileLayer('https://github.com/khufkens/COBECORE_maps/raw/master/cover/{z}/{x}/{y}.png', {
        maxZoom: 16,
	    minZoom: 13,
        tms: false
      }).addTo(map);
      L.control.layers({'Basemap':baselayer},{'orthomosaic':ortho,'forest cover':cover}).addTo(map);
      
function getColor(d) {
    return d == 4  ? '#33a02c' :
           d == 3  ? '#b2df8a' :
           d == 2  ? '#1f78b4' :
           d == 1  ? '#a6cee3' :
                     '#a6cee3' ;
}

var legend = L.control({position: 'bottomright'});

legend.onAdd = function (map) {
      var div = L.DomUtil.create('div', 'info legend'),
         grades = [1, 2, 3, 4],
         labels = ['no change','forest regrowth >1958','forest loss >2000','forest loss >1958'];
    for (var i = 0; i < grades.length; i++) {
        div.innerHTML +=
            '<i style="background:' + getColor(grades[i]) + '"></i> ' +
            labels[i] + '<br>';
    }
    return div;
};
map.addControl(new L.Control.Fullscreen());

legend.addTo(map);

</script>