import React, { Component } from 'react';
import WaveSurfer from 'wavesurfer.js';

export default class HomePage extends Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
        // const aud = document.querySelector('#song');
        this.waveSurfer = WaveSurfer.create({     
            barWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 80,
            progressColor: '#4a74a5',
            responsive: true,
            waveColor: '#ccc',
            cursorWidth: 1,
            cursorColor: '#4a74a5',
        });
        this.waveSurfer.load('http://192.168.50.225:5000/get-wav/flat_normal_double.wav');
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
                </div>
            </div>
        );
    }
}