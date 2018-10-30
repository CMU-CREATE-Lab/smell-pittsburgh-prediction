/*************************************************************************
 * This library is developed by Yen-Chia Hsu
 * Copyright Yen-Chia Hsu.
 * GitHub: https://github.com/yenchiah/geo-heatmap
 * Dependencies: jQuery (http://jquery.com/)
 * Contact: hsu.yenchia@gmail.com
 * License: GNU General Public License v2
 * Version: v1.9
 *************************************************************************/

(function () {
  "use strict";

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Create the class
  //
  var GeoHeatmap = function (container_selector, settings) {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Variables
    //
    var google_map;
    var google_map_data;
    var info_window;
    var highlighted_feature;
    var zipcode_styles = {};

    // Settings
    var init_map_zoom = typeof (settings["init_map_zoom"]) === "undefined" ? 12 : settings["init_map_zoom"];
    var init_map_center = typeof (settings["init_map_center"]) === "undefined" ? {
      lat: 40.43,
      lng: -79.93
    } : settings["init_map_center"];
    var color_opacity = typeof (settings["color_opacity"]) === "undefined" ? 0.7 : settings["color_opacity"];

    // The d3.js color scale object for rendering the geo heatmap
    // IMPORTANT: this feature requires d3.js (https://d3js.org/)
    var color_scale = settings["color_scale"];

    // The max and min percentiles for normalizing the dataset
    // (this is more robust than using min and max values)
    var max_percentile = settings["max_percentile"];
    var min_percentile = settings["min_percentile"];

    // This GeoJSON stores the zipcode boundaries (polygons)
    // (created by using the Python script)
    var zipcode_bound_geoJson = settings["zipcode_bound_geoJson"];

    // This json stores the mapping of zipcode, bounds, and center positions
    // (created by using the Python script)
    var zipcode_bound_info = settings["zipcode_bound_info"];

    // The metadata for visualizing the color of polygons
    var zipcode_metadata = settings["zipcode_metadata"];

    // The function for generating html layout for the info window
    var info_window_html_layout = settings["info_window_html_layout"];

    // Callback functions
    var mouseover_callback = settings["mouseover_callback"];
    var mouseout_callback = settings["mouseout_callback"];
    var info_window_domready_callback = settings["info_window_domready_callback"];
    var info_window_closeclick_callback = settings["info_window_closeclick_callback"];

    // Constants for the zipcode regions
    var ZIPCODE_HIGHLIGHT_STYLE = {
      strokeOpacity: 0.7,
      strokeWeight: 2.5
    };
    var ZIPCODE_HOVER_STYLE = {
      strokeOpacity: 0.7,
      strokeWeight: 2.5
    };

    // Constants for the map
    var MAP_STYLE = [
      {
        featureType: "all",
        stylers: [
          {saturation: -80}
        ]
      }, {
        featureType: "road.arterial",
        elementType: "geometry",
        stylers: [
          {hue: "#00ffee"},
          {saturation: 50}
        ]
      }, {
        featureType: "poi.business",
        elementType: "labels",
        stylers: [
          {visibility: "off"}
        ]
      }
    ];

    // DOM objects
    var $container = $(container_selector);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Private methods
    //
    function init() {
      addMap();
      addZipcodeBoundaries();
    }

    function addMap() {
      // Set map UI
      $container.append($('<div class="geo-heatmap"></div>'));

      // Set Google map
      google_map = new google.maps.Map($(container_selector + " .geo-heatmap").get(0), {
        styles: MAP_STYLE,
        center: init_map_center,
        zoom: init_map_zoom,
        zIndex: 1,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        mapTypeControl: false,
        mapTypeControlOptions: {
          style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
          position: google.maps.ControlPosition.TOP_RIGHT,
          mapTypeIds: [
            google.maps.MapTypeId.HYBRID,
            google.maps.MapTypeId.SATELLITE,
            google.maps.MapTypeId.ROADMAP,
            google.maps.MapTypeId.TERRAIN
          ]
        },
        zoomControl: false,
        zoomControlOptions: {
          position: google.maps.ControlPosition.LEFT_TOP
        },
        streetViewControl: false,
        streetViewControlOptions: {
          position: google.maps.ControlPosition.LEFT_TOP
        },
        scaleControl: true,
        clickableIcons: false
      });
    }

    function addZipcodeBoundaries() {
      // Build a map of zipcodes, features, and styles
      var threshold = 0;
      var zipcode_metadata_nz = normalize(zipcode_metadata, {
        max_percentile: max_percentile,
        min_percentile: min_percentile
      });
      var zipcode_bound_geoJson_features = [];
      for (var i = 0; i < zipcode_bound_geoJson["features"].length; i++) {
        var f = zipcode_bound_geoJson["features"][i];
        var zipcode = f["properties"]["ZCTA5CE10"];
        var fill_color = "#ff0000";
        var metadata, opacity;
        // If no metadata json, use default values
        if (typeof zipcode_metadata === "undefined") {
          metadata = threshold + 1;
          opacity = color_opacity;
        } else {
          metadata = zipcode_metadata[zipcode];
          if (typeof color_scale === "undefined") {
            opacity = zipcode_metadata_nz[zipcode];
          } else {
            opacity = color_opacity;
            fill_color = color_scale(zipcode_metadata_nz[zipcode]);
          }
        }
        // If metadata json exists, need to check if the metadata for a zipcode is undefined
        // Do not show the zipcode that has no metadata or above a threshold
        if (typeof metadata !== "undefined" && metadata > threshold) {
          zipcode_bound_geoJson_features.push(f);
          zipcode_styles[zipcode] = {
            fillColor: fill_color,
            fillOpacity: opacity,
            strokeColor: "#000000",
            strokeOpacity: 0.3,
            strokeWeight: 1
          };
        }
      }
      zipcode_bound_geoJson["features"] = zipcode_bound_geoJson_features;

      // Add GeoJSON to the map as a data layer
      google_map_data = new google.maps.Data();
      var features = google_map_data.addGeoJson(zipcode_bound_geoJson);

      // Add default style
      google_map_data.setStyle(function (feature) {
        var zipcode = feature.getProperty("ZCTA5CE10");
        return zipcode_styles[zipcode];
      });

      // If a zipcode region is highlighted before, highlight the updated one
      if (typeof highlighted_feature !== "undefined") {
        var previous_highlighted_zipcode = highlighted_feature.getProperty("ZCTA5CE10");
        for (var i = 0; i < features.length; i++) {
          f = features[i];
          if (f.getProperty("ZCTA5CE10") === previous_highlighted_zipcode) {
            highlightZipcode({feature: f});
            break;
          }
        }
      }

      // Set the information window for displaying Specks in a polygon on the map
      if (typeof info_window === "undefined") {
        info_window = new google.maps.InfoWindow({
          pixelOffset: new google.maps.Size(0, 0)
        });
        info_window.addListener("domready", function () {
          if (typeof info_window_domready_callback === "function") {
            info_window_domready_callback(this["zipcode"]);
          }
        });
        info_window.addListener("closeclick", function () {
          unhighlightZipcode();
          if (typeof info_window_closeclick_callback === "function") {
            info_window_closeclick_callback(this["zipcode"]);
          }
        });
      }

      // Set click event of the polygons
      google_map_data.addListener("click", function (event) {
        highlightZipcode(event);

        var zipcode = event.feature.getProperty("ZCTA5CE10");
        var html;
        if (typeof info_window_html_layout === "function") {
          html = info_window_html_layout(zipcode);
        } else {
          html = generateDefaultInfoWindowHTML(zipcode);
        }

        var bc = zipcode_bound_info[zipcode];
        var c = new google.maps.LatLng(bc[5], bc[4]);
        info_window["zipcode"] = zipcode;
        info_window.setContent(html);
        info_window.setPosition(c);
        info_window.open(google_map);
      });

      // When the user hovers, tempt them to click by changing color.
      // Call revertStyle() to remove all overrides.
      // This will use the style rules defined in the function passed to setStyle()
      google_map_data.addListener("mouseover", function (event) {
        var zipcode = event.feature.getProperty("ZCTA5CE10");
        var highlighted_zipcode;

        if (typeof highlighted_feature !== "undefined") {
          highlighted_zipcode = highlighted_feature.getProperty("ZCTA5CE10");
        }

        if (zipcode !== highlighted_zipcode) {
          google_map_data.revertStyle();
          google_map_data.overrideStyle(event.feature, ZIPCODE_HOVER_STYLE);
          if (typeof highlighted_feature !== "undefined") {
            google_map_data.overrideStyle(highlighted_feature, ZIPCODE_HIGHLIGHT_STYLE);
          }
        }

        // Callback
        if (typeof mouseover_callback === "function") {
          mouseover_callback(zipcode);
        }
      });

      // Set mouseout event
      google_map_data.addListener("mouseout", function (event) {
        var zipcode = event.feature.getProperty("ZCTA5CE10");
        var highlighted_zipcode;

        if (typeof highlighted_feature !== "undefined") {
          highlighted_zipcode = highlighted_feature.getProperty("ZCTA5CE10");
        }

        if (zipcode !== highlighted_zipcode) {
          google_map_data.revertStyle();
          if (typeof highlighted_feature !== "undefined") {
            google_map_data.overrideStyle(highlighted_feature, ZIPCODE_HIGHLIGHT_STYLE);
          }
        }

        // Callback
        if (typeof mouseout_callback === "function") {
          mouseout_callback(zipcode);
        }
      });

      google_map_data.setMap(google_map);
    }

    function generateDefaultInfoWindowHTML(zipcode) {
      var html = "";
      html += "<table>";
      html += "  <tr>";
      html += "    <td>Zipcode: " + zipcode + "</td>";
      html += "  </tr>";
      if (typeof zipcode_metadata !== "undefined") {
        html += "  <tr>";
        html += "    <td>Metadata: " + zipcode_metadata[zipcode] + "</td>";
        html += "  </tr>";
      }
      html += "</table>";
      return html;
    }

    // Normalize the values in a dictionary
    function normalize(dict, options) {
      if (typeof dict === "undefined") return {};
      if (typeof options === "undefined") options = {};

      var lambda = options["lambda"];
      var max_percentile = typeof options["max_percentile"] === "undefined" ? 1 : options["max_percentile"];
      var min_percentile = typeof options["min_percentile"] === "undefined" ? 0 : options["min_percentile"];
      var desired_max = typeof options["desired_max"] === "undefined" ? 1 : options["desired_max"];
      var desired_min = typeof options["desired_min"] === "undefined" ? 0 : options["desired_min"];
      var dict_trans = typeof lambda === "undefined" ? $.extend({}, dict) : powerTransform(dict, lambda);

      // Normalization
      var dict_nz = {};
      var values = Object.values(dict_trans);
      var max = percentile(values, max_percentile);
      var min = percentile(values, min_percentile);
      var z = max - min;
      for (var key in dict_trans) {
        var tmp = (dict_trans[key] - min) / z;
        // Cap the result in range [desired_min, desired_max]
        if (tmp > desired_max) {
          tmp = desired_max;
        } else if (tmp < desired_min) {
          tmp = desired_min
        }
        dict_nz[key] = tmp;
      }

      return dict_nz;
    }

    function computePercentageWithinStd(arr, n_sigma) {
      // Compute mean
      var sum = 0;
      arr.forEach(function (x) {
        sum += x;
      });
      var mean = sum / arr.length;

      // Compute std
      var res_sq = 0;
      arr.forEach(function (x) {
        res_sq += Math.pow((x - mean), 2);
      });
      var std = Math.pow(res_sq / (arr.length - 1), 0.5);

      // Compute z-score
      var n = 0;
      arr.forEach(function (x) {
        if (Math.abs((x - mean) / std) <= n_sigma) n++;
      });
      return n / arr.length;
    }

    function powerTransform(dict, lambda) {
      var dict_trans = {};

      // Compute geometric mean
      var product = 1;
      var count = 0;
      for (var key in dict) {
        var x = dict[key];
        if (x > 0) {
          product *= x;
          count += 1;
        }
      }
      var gm = Math.pow(product, 1 / count);

      // Transform data
      for (var key in dict) {
        var x = dict[key];
        if (lambda === 0) {
          dict_trans[key] = x > 0 ? gm * Math.log(x) : 0;
        } else {
          dict_trans[key] = x > 0 ? (Math.pow(x, lambda) - 1) / (lambda * Math.pow(gm * Math.log(x), lambda - 1)) : 0;
        }
      }

      return dict_trans;
    }

    function highlightZipcode(event, zoom) {
      var zipcode = event.feature.getProperty("ZCTA5CE10");

      if (zoom === true) {
        // Update current view when a GeoJSON polygon is clicked
        // Get and fit bounds
        var bc = zipcode_bound_info[zipcode];
        var b = new google.maps.LatLngBounds(
          new google.maps.LatLng(bc[1], bc[0]),
          new google.maps.LatLng(bc[3], bc[2])
        );
        google_map.fitBounds(b);
      }

      // Handle style of the data layer
      google_map_data.revertStyle();
      google_map_data.overrideStyle(event.feature, ZIPCODE_HIGHLIGHT_STYLE);
      highlighted_feature = event.feature;
    }

    function unhighlightZipcode() {
      google_map_data.revertStyle();
      highlighted_feature = undefined;
    }

    function percentile(arr, Q) {
      if (arr.length === 1) return arr[0];

      var a = $.merge([], arr);

      a.sort(function (a, b) {
        return a - b
      });

      var q = Math.abs(Q);
      var result = NaN;
      if (q === 0) {
        result = a[0];
      } else if (q === 1) {
        result = a[a.length - 1];
      } else if (q > 0 && q < 1) {
        var i = q * (a.length - 1);
        var j = Math.floor(i);
        result = a[j] + (a[j + 1] - a[j]) * (i - j);
      }

      return Q >= 0 ? result : -result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Privileged methods
    //
    var setZipcodeMetadata = function (desired_zipcode_metadata) {
      zipcode_metadata = desired_zipcode_metadata;
      google_map_data.setMap(null);
      addZipcodeBoundaries();
    };
    this.setZipcodeMetadata = setZipcodeMetadata;

    var setToDefaultView = function () {
      google_map.panTo(init_map_center);
      google_map.setZoom(init_map_zoom);
    };
    this.setToDefaultView = setToDefaultView;

    var getGoogleMap = function () {
      return google_map;
    };
    this.getGoogleMap = getGoogleMap;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Constructor
    //
    init();
  };

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Register to window
//
  if (window.edaplotjs) {
    window.edaplotjs.GeoHeatmap = GeoHeatmap;
  } else {
    window.edaplotjs = {};
    window.edaplotjs.GeoHeatmap = GeoHeatmap;
  }
})();