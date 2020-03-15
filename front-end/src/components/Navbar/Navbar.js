import React from 'react'
import HomeOutlinedIcon from '@material-ui/icons/HomeOutlined';
import './styles.css'

export default class Navbar extends React.Component{
    render(){
        return(
            <div className="homeIcon">
                <a href="/home"><HomeOutlinedIcon/></a>
            </div>
        );
    }
}