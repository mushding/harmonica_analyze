import React from 'react';
import { Redirect, Route } from 'react-router-dom';
import HomePage from './HomePage/HomePage'

function App() {
    return (
		<Route path="/"><HomePage/></Route>
    );
}

export default App;
