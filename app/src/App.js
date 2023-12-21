import './App.css';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import ParallelCoordinatesPlot from './components/ParallelCoordinatesPlot';
// import { data } from './data';
import { useState, useEffect } from 'react';
import axios from 'axios';


function App() {

  const [ selectedModel, setSelectedModel ] = useState("");
  const [ pdata, setPData ] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
       try {
         const response = await axios.get(`http://127.0.0.1:5000/${selectedModel}`);
         setPData(response.data || []);
       } catch (error) {
         console.error('Error fetching data:', error);
       }
    };

    fetchData();
  }, [selectedModel]);


  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const data = [{"name": "image_name", "probabilities": pdata}];


  const [value1, setValue1] = useState(0);
  const [value2, setValue2] = useState(20);

  const handleSliderChange = (slider, event) => {
      const value = event.target.value;
      if (slider === 1) {
        setValue1(value);
      } else if (slider === 2) {
        setValue2(value);
      }
  };

  return (
    <div className="App">
   
      <div className="LeftPanel">

        <div className="ModelSelect">
          <label> Models </label> <br/><br/>
          <select onChange={handleModelChange} value={selectedModel}>
              <option value="">Select Model</option>
              <option value="fetchAlex">AlexNet</option>
              <option value="fetchMobile">MobileNet</option>
          </select>
        </div>

        
        <div className="FromRange">
            <label>From:</label>
            <input
            type="range"
            min="0"
            max="980"
            value={value1}
            onChange={(e) => handleSliderChange(1, e)}
            />
            <span>{value1}</span>
        </div>

        <div className="ToRange">
            <label>To:</label>
            <input
            type="range"
            min="20"
            max="1000"
            value={value2}
            onChange={(e) => handleSliderChange(2, e)}
            />
            <span>{value2}</span>
        </div>
        
      </div>


      <BrowserRouter>
        <Routes>
          <Route
            path='/'
            element={
              <ParallelCoordinatesPlot
                data={data}
                width={1800}
                height={400}
                FROM_VARIABLES={value1}
                TO_VARIABLES={value2}
              />
            }
          />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
