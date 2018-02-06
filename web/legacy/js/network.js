function init() {
  var svg = d3.select("svg");
  var width = +svg.attr("width");
  var height = +svg.attr("height");

  var color = d3.scaleOrdinal(d3.schemeCategory20);

  var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function (d, i) {
      return i;
      //return d.id;
    }).distance(600))
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(width / 2, height / 2));

  var focus_node = null, highlight_node = null;
  var highlight_color = "rgb(31,119,180)";
  var highlight_trans = 0.1;
  var nominal_base_node_size = 8;
  var outline = false;
  var default_node_color = "#ccc";
  var nominal_stroke = 1.5;
  var default_link_color = "#888";

  d3.json("data/network.json", function (error, graph) {
    if (error) throw error;

    var link = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graph.links)
      .enter().append("line")
      .attr("stroke-width", function (d) {
        return Math.sqrt(d.value);
      });

    var node = svg.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("r", 10)
      .attr("fill", function (d) {
        return color(d.group);
      })
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    node.on("mouseover", function (d) {
      set_highlight(d);
    }).on("mouseout", function (d) {
      exit_highlight();
    }).on("mousedown", function (d) {
      console.log("mousedown");
      d3.event.stopPropagation();
      focus_node = d;
      set_focus(d);
      if (highlight_node === null) set_highlight(d);
    });

    d3.select(window).on("mouseup", function () {
      console.log("mouseup");
      if (focus_node !== null) {
        focus_node = null;
        if (highlight_trans < 1) {
          //circle.style("opacity", 1);
          label.style("opacity", 1);
          link.style("opacity", 1);
        }
      }
      if (highlight_node === null) exit_highlight();
    });

    node.append("title")
      .attr("dx", 12)
      .attr("dy", ".35em")
      .text(function (d) {
        return d.id;
      });

    var label = svg.append("g")
      .attr("class", "label")
      .selectAll("text")
      .data(graph.nodes)
      .enter().append("text")
      .attr("dx", 14)
      .attr("dy", ".35em")
      .style("font-size", 14)
      .style("font-family", "Arial, sans-serif")
      .text(function (d) {
        return d.id;
      });

    simulation.nodes(graph.nodes).on("tick", function () {
      link.attr("x1", function (d) {
        return d.source.x;
      }).attr("y1", function (d) {
        return d.source.y;
      }).attr("x2", function (d) {
        return d.target.x;
      }).attr("y2", function (d) {
        return d.target.y;
      });
      node.attr("cx", function (d) {
        return d.x;
      }).attr("cy", function (d) {
        return d.y;
      });
      label.attr("x", function (d) {
        return d.x;
      }).attr("y", function (d) {
        return d.y;
      });
    });

    simulation.force("link").links(graph.links);

    var tocolor = "fill";
    var towhite = "stroke";
    if (outline) {
      tocolor = "stroke";
      towhite = "fill";
    }

    var linkedByIndex = {};
    graph.links.forEach(function (d) {
      linkedByIndex[d.source + "," + d.target] = true;
    });

    function isConnected(a, b) {
      return linkedByIndex[a.index + "," + b.index] || linkedByIndex[b.index + "," + a.index] || a.index == b.index;
    }

    function set_highlight(d) {
      svg.style("cursor", "pointer");
      if (focus_node !== null) d = focus_node;
      highlight_node = d;

      if (highlight_color != "white") {
        /*circle.style(towhite, function (o) {
         return isConnected(d, o) ? highlight_color : "white";
         });*/
        label.style("font-weight", function (o) {
          return isConnected(d, o) ? "bold" : "normal";
        });
        link.style("stroke", function (o) {
          return o.source.index == d.index || o.target.index == d.index ? highlight_color : ((isNumber(o.score) && o.score >= 0) ? color(o.score) : default_link_color);
        });
      }
    }

    function exit_highlight() {
      highlight_node = null;
      if (focus_node === null) {
        svg.style("cursor", "default");
        if (highlight_color != "white") {
          //circle.style(towhite, "white");
          label.style("font-weight", "normal");
          link.style("stroke", function (o) {
            return (isNumber(o.score) && o.score >= 0) ? color(o.score) : default_link_color
          });
        }
      }
    }

    function set_focus(d) {
      if (highlight_trans < 1) {
        /*circle.style("opacity", function (o) {
         return isConnected(d, o) ? 1 : highlight_trans;
         });*/
        label.style("opacity", function (o) {
          return isConnected(d, o) ? 1 : highlight_trans;
        });
        link.style("opacity", function (o) {
          return o.source.index == d.index || o.target.index == d.index ? 1 : highlight_trans;
        });
      }
    }

    function dragstarted(d) {
      if (!d3.event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(d) {
      d.fx = d3.event.x;
      d.fy = d3.event.y;
    }

    function dragended(d) {
      if (!d3.event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    function isNumber(n) {
      return !isNaN(parseFloat(n)) && isFinite(n);
    }
  });
}

