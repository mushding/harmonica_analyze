import React from 'react';
import { Redirect, Route, BrowserRouter, Switch } from 'react-router-dom';
import HomePage from './HomePage/HomePage'

function App() {
    return (
        <BrowserRouter>
            <Switch>
                <Route path="/"><HomePage/></Route>
            </Switch>
        </BrowserRouter>
    );
}

export default App;
