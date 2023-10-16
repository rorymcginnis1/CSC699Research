fetch('Alex_Net_values.json')
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(jsonData => {

    console.log(jsonData);


    let tValueCount = 0;
    let fValueCount = 0;


    jsonData.data.forEach(entry => {
        if (entry.Alex_Net_accuracy) {
            tValueCount += 1;
        } else {
            fValueCount += 1;
        }
    });

    console.log('tValue Count:', tValueCount);
    console.log('fValue Count:', fValueCount);



const data = [tValueCount,fValueCount]
const div = d3.select('#pie-chart-container')
const width = window.innerWidth
const height = window.innerHeight
const radius = Math.min(width,height)/8
const colorScale =d3.scaleOrdinal(['#7326AB','#2A59A9', '#E5A1D4'])
const labels = ['Truth', 'False'];
const avg = div
    .append('svg')
    .attr('width',width)
    .attr('height', height)
    .append('g')
    .attr('transform',`translate(${width/3},${height/2})`)

const pie = d3.pie().value(d =>d).sort(null)
const arc = d3.arc().outerRadius(radius).innerRadius(0)

const g = avg.selectAll('.arc')
    .data(pie(data))
    .enter().append('g')
    .attr('class','arc')
g.append('path')
    .attr('d',arc)
    .attr('class', 'arc')
    .style('fill', (d,i) => colorScale(i))
    .style('stroke','#11141C')
    .style('stroke-width',4)
g.append('text')
    .attr('transform', (d) => `translate(${arc.centroid(d)})`)
    .attr('dy', '0.35em')
    .style('text-anchor', 'middle')
    .text((d, i) => labels[i]);


    

const bdata = [tValueCount, fValueCount];
        const margin = { top: 20, right: 20, bottom: 30, left: 40 };
        const bwidth = 400;
        const bheight = 200;
        
        const bsvg = d3.select('#bar-chart-container')
          .append('svg')
          .attr('width', bwidth)
          .attr('height', bheight);
        
        const xScale = d3.scaleBand()
          .domain(['Category 1', 'Category 2'])
          .range([margin.left, bwidth - margin.right])
          .padding(0.1);
        
        const yScale = d3.scaleLinear()
          .domain([0, d3.max(bdata)])
          .nice()
          .range([bheight - margin.bottom, margin.top]);
        
          bsvg.selectAll('rect')
          .data(bdata)
          .enter()
          .append('rect')
          .attr('x', (d, i) => xScale(`Category ${i + 1}`))
          .attr('y', (d) => yScale(d))
          .attr('width', xScale.bandwidth())
          .attr('height', (d) => bheight - margin.bottom - yScale(d))
          .attr('fill', 'steelblue');
        

        bsvg.append('g')
          .attr('class', 'x-axis')
          .attr('transform', `translate(0, ${bheight - margin.bottom})`)
          .call(d3.axisBottom(xScale));
        

        bsvg.append('g')
          .attr('class', 'y-axis')
          .attr('transform', `translate(${margin.left}, 0)`)
          .call(d3.axisLeft(yScale));


     })
  .catch(error => console.error('Error fetching data:', error.message, error));       