var theData = [1,2,3];

var p = d3.select("body").selectAll("p").data(theData).enter().append("p").text("hello");

//var bodySelection = d3.select("body");

//var svgSelection = bodySelection.append("svg")
	.attr("width", 50)
	.attr("height", 50);

//var circleSelection = svgSelection.append("circle")
	.attr("cx", 25)
	.attr("cy", 25)
	.attr("r", 25)
	.style("fill", "yellow");
