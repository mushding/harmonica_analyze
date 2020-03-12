import React from 'react';
import { Redirect, Route, BrowserRouter, Switch } from 'react-router-dom';
import HomePage from './HomePage/HomePage'

function App() {
    return (
        <BrowserRouter>
            <Switch>
                <Route path="/" render={() => <HomePage/>}/>
            </Switch>
        </BrowserRouter>
    );
}

export default App;
