import React, { Component } from 'react';
import WaveSurfer from 'wavesurfer.js';
import CursorPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.cursor';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions'
import ShowWave from '../../components/ShowWave/ShowWave'
export default class HomePage extends Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
    }
    render(){
        return (
            <div>
                <ShowWave/>
                <h1>TEST</h1>
            </div>
        );
    }
}