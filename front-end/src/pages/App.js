// /src/App.js
import React, { Component } from 'react';
import { HashRouter, Route, Redirect, Switch } from 'react-router-dom';
import HomePage from './HomePage/HomePage';
import StartPage from './StartPage/StartPage'
import RecordPage from './RecordPage/RecordPage'
class App extends Component {
    render() {
        return (
            <HashRouter>
                <Switch>
                    <Route exact path="/" render={() => <Redirect to="/home"/>} />
                    <Route path="/home" component={HomePage} />
                    <Route path="/start" component={StartPage} />
                    <Route path="/record" component={RecordPage} />
                </Switch>
            </HashRouter>
        );
    }
}

export default App;