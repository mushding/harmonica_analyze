import React from 'react';
import WaveSurfer from 'wavesurfer.js';
import CursorPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.cursor';
import RegionsPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.regions';
import MinimapPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.minimap';
import TimelinePlugin from 'wavesurfer.js/dist/plugin/wavesurfer.timeline';
import './styles.css';
import RegionMessage from '../RegionMessage/RegionMessage'
import Collapse from '@material-ui/core/Collapse';
import ColorCircularProgress from '../ProgressCircle/ProgressCircle'
import {
    PlayArrow,
    Pause,
    SyncDisabledOutlined,
    SyncOutlined,
    Refresh,
} from '@material-ui/icons'

export default class ShowWave extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            regionArr: [],
            regionContent: {},
            buttonIsPlay: false,
            buttonIsLoop: false,
            buttonIsShowRegions: true,
            buttonIsShowMiniMap: true,
            buttonIsShowTimeline: true,
            buttonIsShowCursor: true,
            buttonRegionClick: false,
            checked_1: false,
            checked_2: false,
            progressState: true,
        }
    }
    handdleRegionClick = (region) => {
        console.log(region)
        let that = this
		let delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve, s); 
            });
        };
        delay().then(() => {
            that.setState({ buttonRegionClick: false })
            return delay(1000); 
        }).then(() => {
            that.setState({ regionContent: region })
        }).then(() => {
            that.setState({ buttonRegionClick: true })
        })
    }
    initWavesurfer = () => {
        const url = "http://192.168.50.225:5000/wav/" + String(this.props.file)
        this.createWavesurfer()
        if (this.state.regionArr.length === 0) {
            this.createRegions()
        } else {
            this.reCreateRegions()
        }
        this.wavesurfer.load(url)
    }
    createWavesurfer = () => {
        const options = {
            barWidth: 1,
            cursorWidth: 1,
            container: '#waveform',
            backend: 'MediaElement',
            height: 200,
            progressColor: '#005266',
            responsive: true,
            waveColor: '#fff',
            cursorColor: '#005266',
            plugins: [
                RegionsPlugin.create(),
            ]
        };
        const wavesurfer = WaveSurfer.create(options);
        this.wavesurfer = wavesurfer
        this.createMinimap()
        this.createTimeline()
        this.createCursor()

        // event listener
        wavesurfer.on('region-in', (region) => {
            this.region = region
        })
        wavesurfer.on('region-out', this.onPlayEnd)
        wavesurfer.on('region-click', this.handdleRegionClick)
    }
    onPlayEnd = () => {
        this.state.buttonIsLoop ? this.wavesurfer.play() : this.wavesurfer.play(this.region.start)
    }
    createRegions = () => {
        fetch("http://192.168.50.225:5000/getoutput/" + this.props.file)
            .then((response) => {
                let data_promise = Promise.resolve(response.json())
                data_promise.then((data) => {
                    return data
                }).then((jsonData) => {
                    this.setState({ progressState: false })
                    let len = jsonData.length
                    let regionColor = undefined
                    let regionStateArr = []
                    for (let i = 0; i < len; i++){
                        const regionData = jsonData[i]
                        const errorType = jsonData[i]["type"]
                        switch (errorType){
                            case "1":
                                regionColor = "hsla(197, 40%, 23%, 0.3)"
                                break;
                            case "2":
                                regionColor = "hsla(195, 63%, 23%, 0.2)"
                                break;
                        }
                        const regionDir = {
                            start: regionData["start"],
                            end: regionData["end"],
                            drag: regionData["drag"],
                            resize: regionData["resize"],
                            attributes: regionData["type"],
                            color: regionColor,
                        }
                        regionStateArr.push(regionDir)
                        this.wavesurfer.addRegion(regionDir) 
                    }
                    this.setState({ regionArr: regionStateArr})
                })
            })
            .catch((error) => {
                console.log(error)
                return false
            })
        return true
    }
    reCreateRegions = () => {
        let arrLen = this.state.regionArr.length
        for(let i = 0; i < arrLen; i++){
            this.wavesurfer.addRegion(this.state.regionArr[i]) 
        }
    }
    reCreateWavesurfer = () => {
        this.wavesurfer.destroy()
        this.initWavesurfer()
        this.setState({ buttonRegionClick: false })
    }
    createMinimap = () => {
        this.wavesurfer.addPlugin(MinimapPlugin.create({
            container: '#wave-minimap',
            waveColor: '#777',
            progressColor: '#222',
            height: 50,
        })).initPlugin('minimap')
    }
    createTimeline = () => {
        this.wavesurfer.addPlugin(TimelinePlugin.create({
            container: "#wave-timeline",
        })).initPlugin('timeline')
    }
    createCursor = () => {
        this.wavesurfer.addPlugin(CursorPlugin.create({
            showTime: true,
            opacity: 1,
            hideOnBlur: true,
            color: "#005266",
            customShowTimeStyle: {
                'background-color': '#005266',
                color: '#fff',
                padding: '3px',
                'font-size': '15px'
            }
        })).initPlugin('cursor')
    }
    componentDidMount(){
        this.initWavesurfer()
        var that = this;
		setTimeout(() => {
			that.show();
		}, that.props.wait);
    }
    show = () => {
        let that = this
		let delay = (s) => {
            return new Promise((resolve, reject) => {
                setTimeout(resolve, s); 
            });
        };
        delay().then(() => {
            that.setState({checked_1: true})
            return delay(800); 
        }).then(() => {
            that.setState({checked_2: true})
        });
    }
    togglePlay = () => {
        const bool = this.state.buttonIsPlay
        this.wavesurfer.playPause();
        this.setState({ buttonIsPlay: !bool })
    }
    toggleClearRegions = () => {
        const bool = this.state.buttonIsShowRegions
        bool ? this.wavesurfer.clearRegions() : this.reCreateRegions()
        this.setState({ buttonIsShowRegions: !bool })
    }
    toggleLoop = () => {
        const bool = this.state.buttonIsLoop
        this.setState({ buttonIsLoop: !bool })
    }
    toggleShowMiniMap = () => {
        const bool = this.state.buttonIsShowMiniMap
        bool ? this.wavesurfer.destroyPlugin('minimap') : this.createMinimap()
        this.setState({ buttonIsShowMiniMap: !bool})
    }
    toggleShowTimeline = () => {
        const bool = this.state.buttonIsShowTimeline
        bool ? this.wavesurfer.destroyPlugin('timeline') : this.createTimeline()
        this.setState({ buttonIsShowTimeline: !bool})
    }
    toggleShowCursor = () => {
        const bool = this.state.buttonIsShowCursor
        bool ? this.wavesurfer.destroyPlugin('cursor') : this.createCursor()
        this.setState({ buttonIsShowCursor: !bool})
    }
    render(){
        return (
            <div>
                <Collapse in={this.state.checked_1} timeout={1300}> 
                    <div className="waveCursorContainer">
                        <div className="waveContainer">
                            <div id='waveform'></div>
                            <br/>
                            <div id='wave-minimap'></div>
                            <div id='wave-timeline'></div>
                        </div>
                    </div>
                </Collapse>
                <Collapse in={this.state.checked_2} timeout={2000}>
                    <div className="buttonContainer">
                        <button onClick={this.togglePlay} disabled={this.state.progressState}>
                            <div className="buttonIcon">
                                { this.state.buttonIsPlay ? <Pause/> : <PlayArrow/> }
                            </div>
                        </button>
                        <button onClick={this.toggleLoop} disabled={this.state.progressState}>
                            <div className="buttonIcon">
                                { this.state.buttonIsLoop ? <SyncOutlined/> : <SyncDisabledOutlined/> }
                            </div>
                        </button>
                        <button onClick={this.toggleClearRegions} disabled={this.state.progressState}>
                            { this.state.buttonIsShowRegions ? "disable regions" : "able regions" }
                        </button>
                        <button onClick={this.toggleShowMiniMap} disabled={this.state.progressState}>
                            { this.state.buttonIsShowMiniMap ? "close minimap" : "open minimap" }
                        </button>
                        <button onClick={this.toggleShowTimeline} disabled={this.state.progressState}>
                            { this.state.buttonIsShowTimeline ? "close timeline" : "open timeline" }
                        </button>
                        <button onClick={this.toggleShowCursor} disabled={this.state.progressState}>
                            { this.state.buttonIsShowCursor ? "close cursor" : "open sursor"}
                        </button>
                        <button onClick={this.reCreateWavesurfer} disabled={this.state.progressState}>
                            <div className="buttonIcon">
                                <Refresh/>
                            </div>
                        </button>
                        { this.state.progressState  ? (
                            <div className="progressContainer">
                                <ColorCircularProgress/>
                            </div>
                        ) : (
                            <Collapse in={this.state.progressState} timeout={2000}>
                                <div></div>
                            </Collapse>
                        ) }
                    </div>
                </Collapse>
                <Collapse in={this.state.buttonRegionClick} timeout={1000}>
                    <RegionMessage region={this.state.regionContent}/>
                </Collapse>
               
            </div>
        );
    }
}