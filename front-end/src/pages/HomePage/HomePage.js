import React, { Component } from 'react';
import WaveSurfer from 'wavesurfer.js';

export default class HomePage extends Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
        const aud = document.querySelector('#song');
        const params = {
            barWidth: 1,
            cursorWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 80,
            progressColor: '#4a74a5',
            responsive: true,
            waveColor: '#ccc',
            cursorColor: '#4a74a5',
        };
        this.waveSurfer = WaveSurfer.create(params);
        this.waveSurfer.load(aud);
    }
    playIt = () => {
        this.waveSurfer.playPause();
    }
    render(){
        return (
            <div>
                <button onClick={this.playIt}>
                    Play
                </button>
                <div id='waveform'>
                    <audio id='song' src="https://reelcrafter-east.s3.amazonaws.com/aux/test.m4a"/>
                </div>
            </div>
        );
    }
}