import React from 'react'
import './styles.css'
export default class RegionMessage extends React.Component{
    componentDidMount(){
    }
    render(){
        const region = this.props.region
        return(
            <div className="messageContainer">
                <div className="textContainer">
                    { region.attributes === "1" ? (
                        <h3>The Error is have a flat note</h3>
                    ) : region.attributes === "2" ? (
                        <h3>The Error is have a double note</h3>
                    ) : (
                        <h3>attributes: {region.attributes}</h3>
                    )}
                </div>
                <div className="textContainer">
                    <h3>start at {region.start} sec</h3>
                </div>
                <div className="textContainer">
                    <h3>end at {region.end} sec</h3>
                </div>
            </div>
        );
    }
}