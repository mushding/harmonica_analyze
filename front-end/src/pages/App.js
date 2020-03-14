// /src/App.js
import React, { Component } from 'react';
import { BrowserRouter, Redirect, Route } from 'react-router-dom';
import HomePage from './HomePage/HomePage';

class App extends Component {
    render() {
        return (
            <BrowserRouter>
                <Route exact path="/" render={() => <Redirect to="/home"/>} />
                <Route exact path="/home" render={() => <HomePage/>} />
            </BrowserRouter>
            );
        }
}

export default App;