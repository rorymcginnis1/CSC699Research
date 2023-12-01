import './App.css';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import ParallelCoordinatesPlot from './components/ParallelCoordinatesPlot';
import { data } from './data';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route
            path='/'
            element={
              <ParallelCoordinatesPlot
                data={data}
                width={1000}
                height={400}
                VARIABLES={21}
              />
            }
          />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
